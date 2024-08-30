use std::{cell::{Cell, UnsafeCell}, collections::BTreeMap, iter::Peekable, path::PathBuf, rc::Rc};

use rnix::{
    ast::{
        AstToken, Attr, AttrSet, Attrpath, AttrpathValue, Expr, HasEntry, Inherit, InterpolPart,
    },
    TextRange,
};
// NOTE: Importing this from rowan works but not from rnix...
use rowan::ast::AstNode;

use crate::{compiler::Executable, perfstats::measure_ir_generation_time, runnable::Runnable, UnpackedValue, Value};

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

#[derive(Debug, Clone)]
pub(crate) enum Parameter {
    Ident(String),
    Pattern {
        entries: Vec<(String, Option<Rc<Program>>)>,
        binding: Option<String>,
        ignore_unmatched: bool,
    },
}

#[derive(Debug)]
pub(crate) enum AttrsetValue {
    Load,
    Inherit(usize),
    Childset(CreateValueMap),
    Program(Rc<Program>),
}

// FIXME: Is this truly what we want to do?
//        Not the cleanest thing in the world.
#[derive(Debug)]
pub(crate) struct CreateValueMap {
    pub(crate) rec: bool,
    pub(crate) parents: Vec<Rc<Program>>,
    pub(crate) constant: Vec<(String, AttrsetValue)>,
    pub(crate) dynamic: Vec<(Program, AttrsetValue)>,
}

// TODO: Make Programs live in an Arena

// TODO: Adding an accumulator to the virtual machine would simplify things
#[derive(Debug, Clone)]
pub(crate) enum Operation {
    Push(Value),
    // This is different from Push because this also sets the parent scope of the function
    PushFunction(Parameter, Rc<Program>),

    MapBegin {
        parents: Vec<Rc<Program>>,
        rec: bool,
    },
    ConstantAttrPop(String),
    ConstantAttrLoad(String),
    ConstantAttrLazy(String, Rc<Program>),
    ConstantAttrInherit(Rc<String>, usize),
    DynamicAttrPop,
    DynamicAttrLazy(Rc<Program>),

    // Passing parents here again is slightly unclean
    PushAttrset {
        parents: usize,
    },
    GetAttrConsume {
        components: usize,
        default: Option<Rc<Program>>,
    },
    HasAttrpath(usize),
    PushList,
    ListAppend(Rc<Program>),
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
    And(Rc<Program>),
    Or(Rc<Program>),
    Less,
    LessOrEqual,
    MoreOrEqual,
    More,
    Equal,
    NotEqual,
    Implication(Rc<Program>),
    Apply,
    ScopeEnter {
        parents: usize,
    },
    ScopeWith(Rc<Program>),
    // identifier lookup
    Load(String),
    ScopeLeave,
    IfElse(Rc<Program>, Rc<Program>),
    Invert,
    Negate,
    SourceSpanPush(SourceSpan),
    SourceSpanPop,
}

#[derive(Debug)]
pub struct Program {
    pub(crate) executions: Cell<usize>,
    pub(crate) operations: Vec<Operation>,
    pub(crate) compiled: UnsafeCell<Option<Rc<Runnable<Executable>>>>
}

impl Program {
    fn new() -> Self {
        Self { executions: Cell::new(0), operations: vec![], compiled: UnsafeCell::default() }
    }

    #[inline(always)]
    fn push(&mut self, operation: Operation) {
        self.operations.push(operation);
    }
}

pub struct IRCompiler {
    working_directory: PathBuf,
    source_file: Rc<SourceFile>,
}

type ChildMap = BTreeMap<String, ChildNode>;
enum ChildNode {
    Attrset(AttrsetBuilder),
    Value,
}

struct AttrsetBuilder {
    create: CreateValueMap,
    children: ChildMap,
}

impl AttrsetBuilder {
    const fn new(rec: bool) -> Self {
        Self {
            create: CreateValueMap {
                rec,
                parents: vec![],
                constant: vec![],
                dynamic: vec![],
            },
            children: ChildMap::new(),
        }
    }

    fn add_const_attr(&mut self, key: String, value: AttrsetValue) {
        if self.children.contains_key(&key) {
            panic!("invalid attribrute set construction")
        }
        self.children.insert(key.clone(), ChildNode::Value);
        self.create.constant.push((key, value));
    }

    fn add_attr(&mut self, compiler: &IRCompiler, key: Attr, value: AttrsetValue) {
        match compiler.create_maybe_const_attr(key) {
            Ok(constant) => self.add_const_attr(constant, value),
            Err(program) => {
                self.create.dynamic.push((program, value));
            }
        }
    }

    fn inherit(
        &mut self,
        compiler: &IRCompiler,
        from: Option<Program>,
        attrs: impl IntoIterator<Item = Attr>,
    ) {
        for attr in attrs {
            let value = if from.is_some() {
                AttrsetValue::Inherit(self.create.parents.len())
            } else {
                AttrsetValue::Load
            };

            self.add_attr(compiler, attr, value);
        }
        if let Some(source) = from {
            self.create.parents.push(Rc::new(source));
        }
    }

    fn insert_new_attrset_child(&mut self, key: String, rec: bool) -> &mut AttrsetBuilder {
        let ChildNode::Attrset(next) = self
            .children
            .entry(key)
            .or_insert_with(|| ChildNode::Attrset(AttrsetBuilder::new(rec)))
        else {
            unreachable!()
        };
        next
    }

    fn create_dynamic_subset(
        compiler: &IRCompiler,
        mut subpath: Peekable<impl Iterator<Item = Attr>>,
        value: Expr,
    ) -> AttrsetValue {
        if subpath.peek().is_none() {
            AttrsetValue::Program(compiler.create_program(value))
        } else {
            let mut current = AttrsetBuilder::new(false);
            current.pathvalue(compiler, subpath, value);
            AttrsetValue::Childset(current.build())
        }
    }

    fn pathvalue(
        &mut self,
        compiler: &IRCompiler,
        mut attrpath: Peekable<impl Iterator<Item = Attr>>,
        value: Expr,
    ) {
        let mut current = self;

        while let Some(attr) = attrpath.next() {
            match compiler.create_maybe_const_attr(attr) {
                Ok(constant) if attrpath.peek().is_none() => {
                    match value {
                        Expr::AttrSet(set) => {
                            // NOTE: This ignores the `rec` modifier on subsequent merges but nix does the same:
                            //       https://github.com/NixOS/nix/issues/9020
                            let builder = current
                                .insert_new_attrset_child(constant, set.rec_token().is_some());
                            builder.extend_from(compiler, set);
                        }
                        _ => current.add_const_attr(
                            constant,
                            AttrsetValue::Program(compiler.create_program(value)),
                        ),
                    }
                    break;
                }
                Err(program) => {
                    current.create.dynamic.push((
                        program,
                        Self::create_dynamic_subset(compiler, attrpath, value),
                    ));
                    break;
                }
                Ok(constant) => {
                    current = current.insert_new_attrset_child(constant, false);
                }
            }
        }
    }

    fn build(mut self) -> CreateValueMap {
        for (key, child) in self.children {
            match child {
                ChildNode::Attrset(attrset) => self
                    .create
                    .constant
                    .push((key, AttrsetValue::Childset(attrset.build()))),
                ChildNode::Value => (),
            }
        }
        self.create
    }

    fn build_whole(
        compiler: &IRCompiler,
        rec: bool,
        inherits: impl IntoIterator<Item = Inherit>,
        values: impl IntoIterator<Item = AttrpathValue>,
    ) -> CreateValueMap {
        let mut builder = Self::new(rec);
        builder.extend_with(compiler, inherits, values);
        builder.build()
    }

    fn extend_with(
        &mut self,
        compiler: &IRCompiler,
        inherits: impl IntoIterator<Item = Inherit>,
        values: impl IntoIterator<Item = AttrpathValue>,
    ) {
        for inherit in inherits {
            let source = inherit
                .from()
                .and_then(|f| f.expr())
                .map(|source| compiler.create_owned_program(source));
            self.inherit(compiler, source, inherit.attrs());
        }
        for attrvalue in values {
            self.pathvalue(
                compiler,
                attrvalue.attrpath().unwrap().attrs().peekable(),
                attrvalue.value().unwrap(),
            );
        }
    }

    fn extend_from(&mut self, compiler: &IRCompiler, attrset: AttrSet) {
        self.extend_with(compiler, attrset.inherits(), attrset.attrpath_values())
    }
}

impl CreateValueMap {
    fn emit_program(self, program: &mut Program) {
        program.operations.push(Operation::MapBegin {
            parents: self.parents,
            rec: self.rec,
        });

        for (key, value) in self.constant {
            match value {
                AttrsetValue::Load => {
                    program.operations.push(Operation::ConstantAttrLoad(key));
                }
                AttrsetValue::Inherit(i) => {
                    program
                        .operations
                        .push(Operation::ConstantAttrInherit(Rc::new(key), i));
                }
                AttrsetValue::Childset(x) => {
                    let parents = x.parents.len();
                    x.emit_program(program);
                    program.operations.push(Operation::PushAttrset { parents });
                    program.operations.push(Operation::ConstantAttrPop(key));
                }
                AttrsetValue::Program(p) => {
                    program.operations.push(Operation::ConstantAttrLazy(key, p));
                }
            }
        }

        for (key, value) in self.dynamic {
            program.operations.extend(key.operations);
            match value {
                AttrsetValue::Load | AttrsetValue::Inherit(_) => {
                    panic!("dynamic attributes not allowed in inherit");
                }
                AttrsetValue::Childset(x) => {
                    let parents = x.parents.len();
                    x.emit_program(program);
                    program.operations.push(Operation::PushAttrset { parents });
                    program.operations.push(Operation::DynamicAttrPop);
                }
                AttrsetValue::Program(p) => {
                    program.operations.push(Operation::DynamicAttrLazy(p));
                }
            }
        }

        if self.rec {
            program.operations.push(Operation::ScopeLeave);
        }
    }
}

impl IRCompiler {
    fn create_owned_program(&self, expr: Expr) -> Program {
        let mut program = Program::new();
        self.build_program(expr, &mut program);
        program
    }

    fn create_program(&self, expr: Expr) -> Rc<Program> {
        Rc::new(self.create_owned_program(expr))
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

    fn create_maybe_const_string(
        &self,
        parts: Vec<InterpolPart<String>>,
    ) -> Result<String, Program> {
        match &parts[..] {
            [InterpolPart::Literal(_)] => Ok(match parts.into_iter().next() {
                Some(InterpolPart::Literal(x)) => x,
                _ => unreachable!(),
            }),
            _ => {
                let mut program = Program::new();
                self.build_string(parts, |x| x, false, &mut program);
                Err(program)
            }
        }
    }

    fn create_maybe_const_attr(&self, attr: Attr) -> Result<String, Program> {
        match attr {
            Attr::Ident(id) => Ok(id.ident_token().unwrap().green().text().to_string()),
            Attr::Str(string) => self.create_maybe_const_string(string.normalized_parts()),
            Attr::Dynamic(dynamic) => match dynamic.expr().unwrap() {
                Expr::Str(string) => self.create_maybe_const_string(string.normalized_parts()),
                expr => Err(self.create_owned_program(expr)),
            },
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
            Expr::Error(_) => unreachable!(),
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
                rnix::ast::LiteralKind::Uri(x) => {
                    program
                        .operations
                        .push(Operation::Push(UnpackedValue::new_string(
                            x.syntax().text().to_string(),
                        ).pack()))
                }
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
                let map =
                    AttrsetBuilder::build_whole(self, true, x.inherits(), x.attrpath_values());
                let parents = map.parents.len();
                map.emit_program(program);

                program.operations.push(Operation::ScopeEnter { parents });
                self.build_program(x.body().unwrap(), program);
                program.operations.push(Operation::ScopeLeave);
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
                let map = AttrsetBuilder::build_whole(
                    self,
                    x.rec_token().is_some(),
                    x.inherits(),
                    x.attrpath_values(),
                );
                let parents = map.parents.len();
                map.emit_program(program);
                program.operations.push(Operation::PushAttrset { parents });
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
            Expr::With(x) => {
                program.operations.push(Operation::ScopeWith(
                    self.create_program(x.namespace().unwrap()),
                ));
                self.build_program(x.body().unwrap(), program);
                program.operations.push(Operation::ScopeLeave);
            }
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
    ) -> Rc<Program> {
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
