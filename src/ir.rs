use std::{path::PathBuf, rc::Rc};

use rnix::{
    ast::{AstToken, Attr, Attrpath, Expr, HasEntry, InterpolPart},
    TextRange,
};
use rowan::ast::AstNode;

use crate::{perfstats::measure_ir_generation_time, UnpackedValue, Value}; // NOTE: Importing this from rowan works but not from rnix...

#[derive(Debug)]
pub struct SourceFile {
    pub filename: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct SourceSpan {
    // TODO: this doesn't have to be stored per-SourceSpan
    //       (per-Executable instead?)
    pub file: Rc<SourceFile>,
    pub range: TextRange,
}

#[derive(Debug)]
pub enum Parameter {
    Ident(String),
    Pattern {
        entries: Vec<(String, Option<Program>)>,
        binding: Option<String>,
        ignore_unmatched: bool,
    },
}

#[derive(Debug)]
pub enum Operation {
    Push(Value),
    // This is different from Push because this also sets the parent scope of the function
    PushFunction(Parameter, Program),
    CreateAttrset {
        rec: bool,
    },
    InheritAttrs(Option<Program>, Vec<String>),
    SetAttrpath(usize, Program),
    GetAttrConsume {
        components: usize,
        default: Option<Program>,
    },
    HasAttrpath(usize),
    PushList,
    ListAppend(Program),
    StringCloneToMut,
    // TODO: Make construction Operations like StringAppend, SetAttrpath, InheritAttrs etc.
    //       unchecked at runtime (with optional checking for debugging?).
    StringMutAppend,
    StringToPath,
    Concat,
    Update,
    Add,
    Sub,
    Mul,
    Div,
    And(Program),
    Or(Program),
    Less,
    LessOrEqual,
    MoreOrEqual,
    More,
    Equal,
    NotEqual,
    Implication(Program),
    Apply,
    ScopePush,
    ScopeInherit(Option<Program>, Vec<String>),
    ScopeSet(String, Program),
    // identifier lookup
    Load(String),
    ScopePop,
    IfElse(Program, Program),
    Invert,
    Negate,
    SourceSpanPush(SourceSpan),
    SourceSpanPop,
}

#[derive(Debug)]
pub struct Program {
    pub(crate) operations: Vec<Operation>,
}

pub struct IRCompiler {
    working_directory: PathBuf,
    source_file: Rc<SourceFile>,
}

// TODO: Notes on building attrsets
// The reference nix implementation allows using attrpaths with the same prefix like this:
// {
//     a.a = 1;
//     a.b = 1;
// }
// This is simple when there is no dynamic components on the path.
// Fortunately, it is forbidden to do the same with dynamic components.
// This means that the fastest way to compile an attrset build would be to build out a tree
// structure from the attrpaths and probably compile known attributes statically into a base
// attribute set which would then be filled with values and possible dynamic attributes.
// I believe this would improve attrset building performance during runtime pretty significantly

fn attr_assert_const(attr: Attr) -> String {
    match attr {
        Attr::Ident(x) => x.ident_token().unwrap().green().text().to_string(),
        Attr::Dynamic(_) => todo!(),
        Attr::Str(_) => todo!(),
    }
}

impl IRCompiler {
    fn create_program(&self, expr: Expr) -> Program {
        let mut program = Program { operations: vec![] };
        self.build_program(expr, &mut program);
        program
    }

    fn build_attrpath(&self, attrpath: Attrpath, program: &mut Program) -> usize {
        let mut n = 0;
        for attr in attrpath.attrs().collect::<Vec<_>>().into_iter().rev() {
            match attr {
                Attr::Ident(x) => program.operations.push(Operation::Push(
                    UnpackedValue::new_string(x.ident_token().unwrap().green().text().to_string())
                        .pack(),
                )),
                Attr::Dynamic(x) => self.build_program(x.expr().unwrap(), program),
                Attr::Str(x) => self.build_string(x.normalized_parts(), |x| x, false, program),
            };
            n += 1;
        }
        n
    }

    fn build_string<Content>(
        &self,
        parts: impl IntoIterator<Item = InterpolPart<Content>>,
        content_to_text: impl Fn(&Content) -> &str,
        is_path: bool,
        program: &mut Program,
    ) {
        let mut parts = parts.into_iter();
        match parts.next() {
            None => {
                return program.operations.push(Operation::Push(
                    UnpackedValue::new_string(String::new()).pack(),
                ))
            }
            Some(InterpolPart::Literal(literal)) => program.operations.push(Operation::Push(
                UnpackedValue::new_string({
                    let literal = content_to_text(&literal).to_string();
                    if is_path {
                        // not an absolute path
                        if !literal.starts_with("/")
                            && !self.working_directory.as_os_str().is_empty()
                        {
                            format!("{}/{literal}", self.working_directory.to_str().unwrap())
                        } else {
                            literal
                        }
                    } else {
                        literal
                    }
                })
                .pack(),
            )),
            Some(InterpolPart::Interpolation(interpol)) => {
                self.build_program(interpol.expr().unwrap(), program);
            }
        }
        program.operations.push(Operation::StringCloneToMut);
        for part in parts {
            match part {
                InterpolPart::Literal(literal) => program.operations.push(Operation::Push(
                    UnpackedValue::new_string(content_to_text(&literal).to_string()).pack(),
                )),
                InterpolPart::Interpolation(interpol) => {
                    self.build_program(interpol.expr().unwrap(), program);
                }
            }
            program.operations.push(Operation::StringMutAppend);
        }
        if is_path {
            program.operations.push(Operation::StringToPath);
        }
    }

    fn build_program(&self, expr: Expr, program: &mut Program) {
        program
            .operations
            .push(Operation::SourceSpanPush(SourceSpan {
                file: self.source_file.clone(),
                range: expr.syntax().text_range(),
            }));
        match expr {
            Expr::Apply(x) => {
                self.build_program(x.argument().unwrap(), program);
                self.build_program(x.lambda().unwrap(), program);
                program.operations.push(Operation::Apply);
            }
            // TODO:
            Expr::Assert(x) => self.build_program(x.body().unwrap(), program),
            Expr::Error(_) => todo!(),
            Expr::IfElse(x) => {
                self.build_program(x.condition().unwrap(), program);
                let if_true = self.create_program(x.body().unwrap());
                let if_false = self.create_program(x.else_body().unwrap());
                program
                    .operations
                    .push(Operation::IfElse(if_true, if_false))
            }
            Expr::Select(x) => {
                self.build_program(x.expr().unwrap(), program);
                let n = self.build_attrpath(x.attrpath().unwrap(), program);
                program.operations.push(Operation::GetAttrConsume {
                    components: n,
                    default: x.default_expr().map(|expr| self.create_program(expr)),
                })
            }
            Expr::Str(x) => self.build_string(x.normalized_parts(), |x| x, false, program),
            Expr::Path(x) => self.build_string(x.parts(), |x| x.syntax().text(), true, program),
            Expr::Literal(x) => match x.kind() {
                rnix::ast::LiteralKind::Float(x) => program.operations.push(Operation::Push(
                    UnpackedValue::Double(x.value().unwrap()).pack(),
                )),
                rnix::ast::LiteralKind::Integer(x) => {
                    program.operations.push(Operation::Push(
                        UnpackedValue::Integer(x.value().unwrap() as i32).pack(),
                    ));
                }
                rnix::ast::LiteralKind::Uri(_) => todo!(),
            },
            Expr::Lambda(x) => {
                let param = x.param().unwrap();
                let param = match param {
                    rnix::ast::Param::Pattern(pat) => Parameter::Pattern {
                        entries: pat
                            .pat_entries()
                            .map(|x| {
                                (
                                    x.ident().unwrap().to_string(),
                                    x.default().map(|x| self.create_program(x)),
                                )
                            })
                            .collect(),
                        binding: pat.pat_bind().map(|x| x.ident().unwrap().to_string()),
                        ignore_unmatched: pat.ellipsis_token().is_some(),
                    },
                    rnix::ast::Param::IdentParam(ident) => {
                        Parameter::Ident(ident.ident().unwrap().to_string())
                    }
                };
                let body = x.body().unwrap();
                program
                    .operations
                    .push(Operation::PushFunction(param, self.create_program(body)))
            }
            Expr::LegacyLet(_) => todo!(),
            Expr::LetIn(x) => {
                program.operations.push(Operation::ScopePush);
                for inherit in x.inherits() {
                    let source = inherit
                        .from()
                        .and_then(|f| f.expr())
                        .map(|source| self.create_program(source));
                    program.operations.push(Operation::ScopeInherit(
                        source,
                        inherit.attrs().map(attr_assert_const).collect(),
                    ));
                }
                for entry in x.attrpath_values() {
                    let mut it = entry.attrpath().unwrap().attrs();
                    let name = attr_assert_const(it.next().unwrap());
                    let value = entry.value().unwrap();
                    let value_program = self.create_program(value);
                    program
                        .operations
                        .push(Operation::ScopeSet(name, value_program));
                }
                self.build_program(x.body().unwrap(), program);
                program.operations.push(Operation::ScopePop);
            }
            Expr::List(x) => {
                program.operations.push(Operation::PushList);
                for item in x.items() {
                    let value_program = self.create_program(item);
                    program
                        .operations
                        .push(Operation::ListAppend(value_program));
                }
            }
            Expr::BinOp(x) => {
                self.build_program(x.lhs().unwrap(), program);
                match x.operator().unwrap() {
                    rnix::ast::BinOpKind::And
                    | rnix::ast::BinOpKind::Or
                    | rnix::ast::BinOpKind::Implication => (),
                    _ => self.build_program(x.rhs().unwrap(), program),
                };
                program.operations.push(match x.operator().unwrap() {
                    rnix::ast::BinOpKind::Concat => Operation::Concat,
                    rnix::ast::BinOpKind::Update => Operation::Update,
                    rnix::ast::BinOpKind::Add => Operation::Add,
                    rnix::ast::BinOpKind::Sub => Operation::Sub,
                    rnix::ast::BinOpKind::Mul => Operation::Mul,
                    rnix::ast::BinOpKind::Div => Operation::Div,
                    rnix::ast::BinOpKind::Equal => Operation::Equal,
                    rnix::ast::BinOpKind::Less => Operation::Less,
                    rnix::ast::BinOpKind::LessOrEq => Operation::LessOrEqual,
                    rnix::ast::BinOpKind::More => Operation::More,
                    rnix::ast::BinOpKind::MoreOrEq => Operation::MoreOrEqual,
                    rnix::ast::BinOpKind::NotEqual => Operation::NotEqual,
                    rnix::ast::BinOpKind::And => {
                        Operation::And(self.create_program(x.rhs().unwrap()))
                    }
                    rnix::ast::BinOpKind::Or => {
                        Operation::Or(self.create_program(x.rhs().unwrap()))
                    }
                    rnix::ast::BinOpKind::Implication => {
                        Operation::Implication(self.create_program(x.rhs().unwrap()))
                    }
                })
            }
            Expr::Paren(x) => self.build_program(x.expr().unwrap(), program),
            Expr::AttrSet(x) => {
                program.operations.push(Operation::CreateAttrset {
                    rec: x.rec_token().is_some(),
                });
                for inherit in x.inherits() {
                    let source = inherit
                        .from()
                        .and_then(|f| f.expr())
                        .map(|source| self.create_program(source));
                    program.operations.push(Operation::InheritAttrs(
                        source,
                        inherit.attrs().map(attr_assert_const).collect(),
                    ));
                }
                for keyvalue in x.attrpath_values() {
                    let path = keyvalue.attrpath().unwrap();
                    let value = self.create_program(keyvalue.value().unwrap());
                    let n = self.build_attrpath(path, program);
                    program.operations.push(Operation::SetAttrpath(n, value));
                }
            }
            Expr::UnaryOp(x) => {
                self.build_program(x.expr().unwrap(), program);
                program.operations.push(match x.operator().unwrap() {
                    rnix::ast::UnaryOpKind::Invert => Operation::Invert,
                    rnix::ast::UnaryOpKind::Negate => Operation::Negate,
                })
            }
            Expr::Ident(x) => program.operations.push(Operation::Load(
                x.ident_token().unwrap().green().text().to_string(),
            )),
            Expr::With(_) => todo!(),
            Expr::HasAttr(x) => {
                self.build_program(x.expr().unwrap(), program);
                let attrpath = x.attrpath().unwrap();
                let n = self.build_attrpath(attrpath, program);
                program.operations.push(Operation::HasAttrpath(n))
            }
            _ => todo!(),
        }
        program.operations.push(Operation::SourceSpanPop);
    }

    pub fn compile(
        working_directory: PathBuf,
        filename: String,
        file_content: String,
        expression: Expr,
    ) -> Program {
        let _tc = measure_ir_generation_time();
        IRCompiler {
            working_directory,
            source_file: Rc::new(SourceFile {
                filename,
                content: file_content,
            }),
        }
        .create_program(expression)
    }
}
