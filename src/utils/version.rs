use std::{cmp::Ordering, str::Chars};

pub struct VersionComponentIterator<'a> {
    // NOTE: Peekable can't be used here because SourceIter is not stable
    //       also CharIndices would probably have to be used instead of as_str
    chars: Chars<'a>,
    peek: Option<(&'a str, char)>,
}

impl<'a> VersionComponentIterator<'a> {
    pub fn new(chars: Chars<'a>) -> Self {
        Self { chars, peek: None }
    }

    fn populate_peek(&mut self) {
        let str = self.chars.as_str();
        self.peek = self.chars.next().map(|c| (str, c));
    }

    fn inner_peek(&mut self) -> Option<(&'a str, &char)> {
        if self.peek.is_none() {
            self.populate_peek()
        }
        self.peek.as_ref().map(|(a, b)| (*a, b))
    }

    fn inner_next_if(&mut self, predicate: impl FnOnce(&char) -> bool) -> Option<char> {
        if self.peek.is_none() {
            self.populate_peek()
        }
        self.peek.take_if(|x| predicate(&x.1)).map(|x| x.1)
    }
}

impl<'a> Iterator for VersionComponentIterator<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        while self.inner_next_if(|c| *c == '.' || *c == '-').is_some() {}

        let (start, next) = self.inner_peek()?;

        if next.is_ascii_digit() {
            while self.inner_next_if(|x| x.is_ascii_digit()).is_some() {}
        } else {
            while self
                .inner_next_if(|x| !x.is_ascii_digit() && *x != '.' && *x != '-')
                .is_some()
            {}
        };

        Some(match self.inner_peek().map(|x| x.0) {
            Some(end) => &start[..(unsafe { end.as_ptr().offset_from(start.as_ptr()) as usize })],
            None => start,
        })
    }
}

#[test]
fn test_version_component_iterator() {
    macro_rules! make_test {
        ($a: literal, $r: expr) => {
            assert_eq!(
                VersionComponentIterator::new($a.chars())
                    .collect::<Vec<_>>()
                    .as_slice(),
                &$r
            )
        };
    }

    make_test!("1.2.3", ["1", "2", "3"]);
    make_test!("1.2.3-alpha1", ["1", "2", "3", "alpha", "1"]);
    make_test!("unstable-2024-04-10", ["unstable", "2024", "04", "10"]);
    make_test!("--1.-2--", ["1", "2"]);
}

fn version_component_less(a: &str, b: &str) -> bool {
    let a_int = a.parse::<u64>().ok();
    let b_int = b.parse::<u64>().ok();

    #[allow(clippy::if_same_then_else)]
    if let Some((a, b)) = a_int.and_then(|x| b_int.map(|y| (x, y))) {
        a < b
    } else if a.is_empty() && b_int.is_some() {
        true
    } else if a == "pre" && b != "pre" {
        true
    } else if b == "pre" {
        false
    } else if b_int.is_some() {
        true
    } else if a_int.is_some() {
        false
    } else {
        a < b
    }
}

pub fn version_compare(a: &str, b: &str) -> Ordering {
    let mut ait = VersionComponentIterator::new(a.chars());
    let mut bit = VersionComponentIterator::new(b.chars());

    loop {
        let ac = ait.next();
        let bc = bit.next();
        if ac.is_none() && bc.is_none() {
            return Ordering::Equal;
        }
        let astr = ac.unwrap_or_default();
        let bstr = bc.unwrap_or_default();
        if version_component_less(astr, bstr) {
            return Ordering::Less;
        } else if version_component_less(bstr, astr) {
            return Ordering::Greater;
        }
    }
}

#[test]
fn test_version_compare() {
    macro_rules! make_test {
        ($a: literal == $b: literal) => {
            assert_eq!(
                version_compare($a, $b),
                Ordering::Equal
            );
        };
        ($a: literal $op: tt $b: literal) => {
            assert_eq!(
                version_compare($a, $b),
                make_test!(@order $op)
            );
            assert_eq!(
                version_compare($b, $a),
                make_test!(@reverse_order $op)
            );
        };
        (@order >) => { Ordering::Greater };
        (@order <) => { Ordering::Less };
        (@reverse_order >) => { Ordering::Less };
        (@reverse_order <) => { Ordering::Greater };
    }

    make_test!("1.2.3" == "1.2.3");
    // NOTE: This is how it works in nix.
    make_test!("1.2.3-alpha1" > "1.2.3");
    make_test!("unstable-2024-04-10" < "unstable-2024-04-23");
    make_test!("0.1-pre1" < "0.1");
    make_test!("unstable-2024-04-10" < "0.2.3");
}
