# Mutual-friend recommendations

The input is an adjacency list. Each row contains a user ID, a tab, and comma-separated friend IDs. Parsing removes duplicates and self-links. Each friendship becomes two directed edges, so a missing reverse listing does not change the result. Isolated users remain eligible for the sample.

The assignment asks for recommendations for ten users. The implementation chooses those users first, then expands only paths that start at one of them. For a sampled user u, every path u -> m -> v contributes one mutual friend to the pair (u, v). Distinct edges guarantee that a middle user m is counted once. The program removes self-recommendations and existing friendships, then ranks candidates by descending count and ascending numeric ID.

## MapReduce pseudocode

```text
E = DISTINCT(all parsed edges and their reverse edges)
U = DISTINCT(row IDs and edge endpoints)
S = ten users with smallest (SHA256(seed:user), user) keys

MAP each (u, m) in E with u in S to (m, u)
JOIN these records with E keyed by its source
MAP each joined (m, (u, v)) to ((u, v), 1)
FILTER u != v
REDUCE by (u, v), adding counts
SUBTRACT keys corresponding to existing edges in E
MAP ((u, v), count) to (u, [(v, count)])
REDUCE by u, merging lists and retaining the best ten candidates
FOR each user in S:
    print user + TAB + comma-separated candidate IDs
    if no candidates exist, print the user followed by a tab
```

Keeping only ten entries in each partial ranking is exact: a candidate outside a partial top ten already has ten better candidates in that partial list and cannot enter the final top ten. Each ranking accumulator therefore holds at most ten entries.

Let d(u) be a user's degree and S the sampled set. Path expansion emits W = sum over u in S of sum over m in N(u) of d(m) records. Deduplication and the joins scan the edge set; counting processes W paths. This avoids generating all pairs of neighbors for every user in the graph. Expanding paths for every user would emit roughly sum over m of d(m)^2 paths, even though only ten users need recommendations. High-degree intermediate users can still cause skew.

Only the ten chosen IDs and their final rankings are returned to the driver. The adjacency list, joins and counts stay in Spark. Recommendations require at least one mutual friend, so users with no eligible two-hop neighbors receive an empty list.
