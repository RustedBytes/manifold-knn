use core::fmt;

/// Errors returned by `manifold-knn` construction, update, and query APIs.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Number of points and number of successor lists differ.
    TableLengthMismatch {
        /// Number of points.
        points: usize,
        /// Number of successor lists.
        lists: usize,
    },
    /// A point coordinate is `NaN` or infinite.
    NonFiniteCoordinate {
        /// Point index.
        point: usize,
        /// Coordinate index inside the point.
        coordinate: usize,
        /// Invalid coordinate value.
        value: f64,
    },
    /// A query coordinate is `NaN` or infinite.
    NonFiniteQuery {
        /// Coordinate index inside the query.
        coordinate: usize,
        /// Invalid coordinate value.
        value: f64,
    },
    /// Prefix length exceeds the number of stored points.
    InvalidPrefix {
        /// Requested prefix length.
        prefix_len: usize,
        /// Number of stored points.
        len: usize,
    },
    /// A point index is out of bounds.
    InvalidIndex {
        /// Requested index.
        index: usize,
        /// Number of stored points or lists.
        len: usize,
    },
    /// A successor edge is invalid. Successors must satisfy `owner < successor < len`.
    InvalidSuccessor {
        /// Owner list index.
        owner: usize,
        /// Successor index.
        successor: usize,
        /// Number of stored points or lists.
        len: usize,
    },
    /// A successor list is not sorted.
    UnsortedSuccessorList {
        /// Owner list index.
        owner: usize,
        /// Previous entry.
        previous: usize,
        /// Current out-of-order entry.
        current: usize,
    },
    /// A successor list contains a duplicate entry.
    DuplicateSuccessor {
        /// Owner list index.
        owner: usize,
        /// Duplicated successor index.
        successor: usize,
    },
    /// An insertion-time neighbor is not an earlier birth index.
    InvalidInsertionNeighbor {
        /// Index of the point being inserted.
        inserted: usize,
        /// Invalid neighbor index.
        neighbor: usize,
    },
    /// An operation requires an active point, but the point is inactive.
    InactivePoint {
        /// Inactive point index.
        index: usize,
    },
    /// The optional Delaunay backend returned an error.
    DelaunayKernel {
        /// Delaunay operation that failed.
        operation: &'static str,
        /// Backend error message.
        message: String,
    },
    /// The Delaunay backend produced an internally inconsistent state.
    DelaunayInvariant {
        /// Human-readable invariant violation.
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TableLengthMismatch { points, lists } => write!(
                formatter,
                "successor table length mismatch: {points} points but {lists} lists"
            ),
            Error::NonFiniteCoordinate {
                point,
                coordinate,
                value,
            } => write!(
                formatter,
                "point {point} has non-finite coordinate {coordinate}: {value}"
            ),
            Error::NonFiniteQuery { coordinate, value } => write!(
                formatter,
                "query has non-finite coordinate {coordinate}: {value}"
            ),
            Error::InvalidPrefix { prefix_len, len } => write!(
                formatter,
                "invalid prefix length {prefix_len}; index contains {len} points"
            ),
            Error::InvalidIndex { index, len } => {
                write!(formatter, "index {index} out of bounds for length {len}")
            }
            Error::InvalidSuccessor {
                owner,
                successor,
                len,
            } => write!(
                formatter,
                "invalid successor edge {owner} -> {successor}; expected owner < successor < {len}"
            ),
            Error::UnsortedSuccessorList {
                owner,
                previous,
                current,
            } => write!(
                formatter,
                "successor list {owner} is not sorted: {previous} appears before {current}"
            ),
            Error::DuplicateSuccessor { owner, successor } => write!(
                formatter,
                "successor list {owner} contains duplicate successor {successor}"
            ),
            Error::InvalidInsertionNeighbor { inserted, neighbor } => write!(
                formatter,
                "invalid insertion neighbor {neighbor} for inserted point {inserted}; \
expected neighbor < inserted"
            ),
            Error::InactivePoint { index } => write!(formatter, "point {index} is inactive"),
            Error::DelaunayKernel { operation, message } => {
                write!(
                    formatter,
                    "Delaunay backend failed during {operation}: {message}"
                )
            }
            Error::DelaunayInvariant { message } => {
                write!(formatter, "Delaunay backend invariant violation: {message}")
            }
        }
    }
}

impl std::error::Error for Error {}
