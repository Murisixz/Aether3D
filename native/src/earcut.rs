//! Ear clipping of polygons with holes, many polygons per call: the line-by-line twin of
//! `aether3d.cad.triangulate.triangulate_numpy` (see its docstring for the algorithm).
//! Every decision uses the same float64 `+ - * /` in the same order and the same
//! comparisons, and every sort is stable with the same keys, so the triangles are
//! identical (`tests/test_native.py`).

const Z_ORDER_THRESHOLD: usize = 80;

struct Earcut<'a> {
    xs: &'a [f64],
    ys: &'a [f64],
    i: Vec<i64>,
    x: Vec<f64>,
    y: Vec<f64>,
    prev: Vec<i64>,
    next: Vec<i64>,
    z: Vec<i64>,
    pz: Vec<i64>,
    nz: Vec<i64>,
    triangles: Vec<i64>,
    min_x: f64,
    min_y: f64,
    inv_size: f64,
}

fn sign(v: f64) -> i32 {
    (v > 0.0) as i32 - (v < 0.0) as i32
}

#[allow(clippy::too_many_arguments)]
fn in_triangle(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, px: f64, py: f64) -> bool {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0.0
        && (cx - bx) * (py - by) - (cy - by) * (px - bx) >= 0.0
        && (ax - cx) * (py - cy) - (ay - cy) * (px - cx) >= 0.0
}

fn pmin(a: f64, b: f64) -> f64 {
    // Python's min(a, b): the first unless the second is smaller.
    if b < a {
        b
    } else {
        a
    }
}

fn pmax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

impl<'a> Earcut<'a> {
    fn new(xs: &'a [f64], ys: &'a [f64]) -> Self {
        Earcut {
            xs,
            ys,
            i: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            prev: Vec::new(),
            next: Vec::new(),
            z: Vec::new(),
            pz: Vec::new(),
            nz: Vec::new(),
            triangles: Vec::new(),
            min_x: 0.0,
            min_y: 0.0,
            inv_size: 0.0,
        }
    }

    #[inline]
    fn nx(&self, p: i64) -> i64 {
        self.next[p as usize]
    }

    #[inline]
    fn pv(&self, p: i64) -> i64 {
        self.prev[p as usize]
    }

    #[inline]
    fn px(&self, p: i64) -> f64 {
        self.x[p as usize]
    }

    #[inline]
    fn py(&self, p: i64) -> f64 {
        self.y[p as usize]
    }

    #[inline]
    fn vi(&self, p: i64) -> i64 {
        self.i[p as usize]
    }

    fn node(&mut self, vi: i64) -> i64 {
        let n = self.i.len() as i64;
        self.i.push(vi);
        self.x.push(self.xs[vi as usize]);
        self.y.push(self.ys[vi as usize]);
        self.prev.push(n);
        self.next.push(n);
        self.z.push(0);
        self.pz.push(-1);
        self.nz.push(-1);
        n
    }

    fn insert_after(&mut self, vi: i64, last: i64) -> i64 {
        let n = self.node(vi);
        if last >= 0 {
            let nx = self.nx(last);
            self.next[n as usize] = nx;
            self.prev[n as usize] = last;
            self.prev[nx as usize] = n;
            self.next[last as usize] = n;
        }
        n
    }

    fn remove(&mut self, p: i64) {
        let (nx, pv) = (self.nx(p), self.pv(p));
        self.prev[nx as usize] = pv;
        self.next[pv as usize] = nx;
        let (pz, nz) = (self.pz[p as usize], self.nz[p as usize]);
        if pz >= 0 {
            self.nz[pz as usize] = nz;
        }
        if nz >= 0 {
            self.pz[nz as usize] = pz;
        }
    }

    fn ring(&mut self, start: i64, end: i64, ccw: bool) -> i64 {
        let mut area = 0.0f64;
        let mut j = end - 1;
        for k in start..end {
            let (k_, j_) = (k as usize, j as usize);
            area += (self.xs[j_] - self.xs[k_]) * (self.ys[k_] + self.ys[j_]);
            j = k;
        }
        let mut last = -1i64;
        if ccw == (area > 0.0) {
            for k in start..end {
                last = self.insert_after(k, last);
            }
        } else {
            let mut k = end - 1;
            while k > start - 1 {
                last = self.insert_after(k, last);
                k -= 1;
            }
        }
        if last >= 0 && self.equals(last, self.nx(last)) {
            self.remove(last);
            last = self.nx(last);
        }
        last
    }

    #[inline]
    fn cross(&self, a: i64, b: i64, c: i64) -> f64 {
        let (xa, ya) = (self.px(a), self.py(a));
        (self.px(b) - xa) * (self.py(c) - ya) - (self.py(b) - ya) * (self.px(c) - xa)
    }

    #[inline]
    fn equals(&self, a: i64, b: i64) -> bool {
        self.px(a) == self.px(b) && self.py(a) == self.py(b)
    }

    fn locally_inside(&self, a: i64, b: i64) -> bool {
        let (an, ap) = (self.nx(a), self.pv(a));
        if self.cross(ap, a, an) > 0.0 {
            return self.cross(a, an, b) >= 0.0 && self.cross(a, b, ap) >= 0.0;
        }
        self.cross(a, an, b) > 0.0 || self.cross(a, b, ap) > 0.0
    }

    fn middle_inside(&self, a: i64, b: i64) -> bool {
        let mut p = a;
        let mut inside = false;
        let px = (self.px(a) + self.px(b)) / 2.0;
        let py = (self.py(a) + self.py(b)) / 2.0;
        loop {
            let q = self.nx(p);
            if (self.py(p) > py) != (self.py(q) > py) && self.py(q) != self.py(p) {
                let xc = (self.px(q) - self.px(p)) * (py - self.py(p)) / (self.py(q) - self.py(p))
                    + self.px(p);
                if px < xc {
                    inside = !inside;
                }
            }
            p = q;
            if p == a {
                return inside;
            }
        }
    }

    fn on_segment(&self, p: i64, q: i64, r: i64) -> bool {
        let (xp, xq, xr) = (self.px(p), self.px(q), self.px(r));
        let (yp, yq, yr) = (self.py(p), self.py(q), self.py(r));
        pmin(xp, xr) <= xq && xq <= pmax(xp, xr) && pmin(yp, yr) <= yq && yq <= pmax(yp, yr)
    }

    fn intersects(&self, p1: i64, q1: i64, p2: i64, q2: i64) -> bool {
        let o1 = sign(self.cross(p1, q1, p2));
        let o2 = sign(self.cross(p1, q1, q2));
        let o3 = sign(self.cross(p2, q2, p1));
        let o4 = sign(self.cross(p2, q2, q1));
        if o1 != o2 && o3 != o4 {
            return true;
        }
        if o1 == 0 && self.on_segment(p1, p2, q1) {
            return true;
        }
        if o2 == 0 && self.on_segment(p1, q2, q1) {
            return true;
        }
        if o3 == 0 && self.on_segment(p2, p1, q2) {
            return true;
        }
        o4 == 0 && self.on_segment(p2, q1, q2)
    }

    fn intersects_polygon(&self, a: i64, b: i64) -> bool {
        let mut p = a;
        let (ia, ib) = (self.vi(a), self.vi(b));
        loop {
            let q = self.nx(p);
            if self.vi(p) != ia
                && self.vi(q) != ia
                && self.vi(p) != ib
                && self.vi(q) != ib
                && self.intersects(p, q, a, b)
            {
                return true;
            }
            p = q;
            if p == a {
                return false;
            }
        }
    }

    fn valid_diagonal(&self, a: i64, b: i64) -> bool {
        self.vi(self.nx(a)) != self.vi(b)
            && self.vi(self.pv(a)) != self.vi(b)
            && !self.intersects_polygon(a, b)
            && ((self.locally_inside(a, b)
                && self.locally_inside(b, a)
                && self.middle_inside(a, b)
                && (self.cross(self.pv(a), a, self.pv(b)) != 0.0
                    || self.cross(a, self.pv(b), b) != 0.0))
                || (self.equals(a, b)
                    && self.cross(self.pv(a), a, self.nx(a)) < 0.0
                    && self.cross(self.pv(b), b, self.nx(b)) < 0.0))
    }

    fn is_ear(&self, ear: i64) -> bool {
        let (a, b, c) = (self.pv(ear), ear, self.nx(ear));
        if self.cross(a, b, c) <= 0.0 {
            return false;
        }
        let (ax, ay, bx, by, cx, cy) = (
            self.px(a),
            self.py(a),
            self.px(b),
            self.py(b),
            self.px(c),
            self.py(c),
        );
        let mut p = self.nx(c);
        while p != a {
            let (x, y) = (self.px(p), self.py(p));
            if !(x == ax && y == ay)
                && !(x == cx && y == cy)
                && !(x == bx && y == by)
                && in_triangle(ax, ay, bx, by, cx, cy, x, y)
                && self.cross(self.pv(p), p, self.nx(p)) <= 0.0
            {
                return false;
            }
            p = self.nx(p);
        }
        true
    }

    fn is_ear_hashed(&self, ear: i64) -> bool {
        let (a, b, c) = (self.pv(ear), ear, self.nx(ear));
        if self.cross(a, b, c) <= 0.0 {
            return false;
        }
        let (ax, ay, bx, by, cx, cy) = (
            self.px(a),
            self.py(a),
            self.px(b),
            self.py(b),
            self.px(c),
            self.py(c),
        );
        let x0 = pmin(pmin(ax, bx), cx);
        let y0 = pmin(pmin(ay, by), cy);
        let x1 = pmax(pmax(ax, bx), cx);
        let y1 = pmax(pmax(ay, by), cy);
        let min_z = self.z_order(x0, y0);
        let max_z = self.z_order(x1, y1);
        let blocks = |p: i64| -> bool {
            let (x, y) = (self.px(p), self.py(p));
            x0 <= x
                && x <= x1
                && y0 <= y
                && y <= y1
                && p != a
                && p != c
                && !(x == ax && y == ay)
                && !(x == cx && y == cy)
                && !(x == bx && y == by)
                && in_triangle(ax, ay, bx, by, cx, cy, x, y)
                && self.cross(self.pv(p), p, self.nx(p)) <= 0.0
        };
        let mut p = self.pz[ear as usize];
        let mut n = self.nz[ear as usize];
        while p >= 0 && self.z[p as usize] >= min_z && n >= 0 && self.z[n as usize] <= max_z {
            if blocks(p) || blocks(n) {
                return false;
            }
            p = self.pz[p as usize];
            n = self.nz[n as usize];
        }
        while p >= 0 && self.z[p as usize] >= min_z {
            if blocks(p) {
                return false;
            }
            p = self.pz[p as usize];
        }
        while n >= 0 && self.z[n as usize] <= max_z {
            if blocks(n) {
                return false;
            }
            n = self.nz[n as usize];
        }
        true
    }

    fn z_order(&self, px: f64, py: f64) -> i64 {
        // Python's int() truncates toward zero; so does `as` (values here are >= 0).
        let mut xi = ((px - self.min_x) * self.inv_size) as i64;
        let mut yi = ((py - self.min_y) * self.inv_size) as i64;
        xi = (xi | (xi << 8)) & 0x00FF00FF;
        xi = (xi | (xi << 4)) & 0x0F0F0F0F;
        xi = (xi | (xi << 2)) & 0x33333333;
        xi = (xi | (xi << 1)) & 0x55555555;
        yi = (yi | (yi << 8)) & 0x00FF00FF;
        yi = (yi | (yi << 4)) & 0x0F0F0F0F;
        yi = (yi | (yi << 2)) & 0x33333333;
        yi = (yi | (yi << 1)) & 0x55555555;
        xi | (yi << 1)
    }

    fn index_curve(&mut self, start: i64) {
        let mut nodes = Vec::new();
        let mut p = start;
        loop {
            if self.z[p as usize] == 0 {
                self.z[p as usize] = self.z_order(self.px(p), self.py(p));
            }
            nodes.push(p);
            p = self.nx(p);
            if p == start {
                break;
            }
        }
        let z = &self.z;
        nodes.sort_by_key(|&k| z[k as usize]); // stable, like Python's sort
        let m = nodes.len();
        for (k, &nd) in nodes.iter().enumerate() {
            self.pz[nd as usize] = if k > 0 { nodes[k - 1] } else { -1 };
            self.nz[nd as usize] = if k + 1 < m { nodes[k + 1] } else { -1 };
        }
    }

    fn filter(&mut self, start: i64, end: i64, collinear: bool) -> i64 {
        if start < 0 {
            return start;
        }
        let mut end = if end < 0 { start } else { end };
        let mut p = start;
        loop {
            let mut again = false;
            let nx = self.nx(p);
            if self.equals(p, nx) || (collinear && self.cross(self.pv(p), p, nx) == 0.0) {
                self.remove(p);
                p = self.pv(p);
                end = p;
                if p == self.nx(p) {
                    break;
                }
                again = true;
            } else {
                p = nx;
            }
            if !again && p == end {
                break;
            }
        }
        end
    }

    fn filter_duplicates(&mut self, start: i64, end: i64) -> i64 {
        self.filter(start, end, false)
    }

    fn cure_local_intersections(&mut self, start: i64) -> i64 {
        let mut start = start;
        let mut p = start;
        loop {
            let a = self.pv(p);
            let b = self.nx(self.nx(p));
            if !self.equals(a, b)
                && self.intersects(a, p, self.nx(p), b)
                && self.locally_inside(a, b)
                && self.locally_inside(b, a)
            {
                let tri = [self.vi(a), self.vi(p), self.vi(b)];
                self.triangles.extend_from_slice(&tri);
                let pn = self.nx(p);
                self.remove(pn);
                self.remove(p);
                start = b;
                p = b;
            }
            p = self.nx(p);
            if p == start {
                break;
            }
        }
        self.filter_duplicates(p, -1)
    }

    fn split_polygon(&mut self, a: i64, b: i64) -> i64 {
        let a2 = self.node(self.vi(a));
        let b2 = self.node(self.vi(b));
        let (an, bp) = (self.nx(a), self.pv(b));
        self.next[a as usize] = b;
        self.prev[b as usize] = a;
        self.next[a2 as usize] = an;
        self.prev[an as usize] = a2;
        self.next[b2 as usize] = a2;
        self.prev[a2 as usize] = b2;
        self.next[bp as usize] = b2;
        self.prev[b2 as usize] = bp;
        b2
    }

    fn split_earcut(&mut self, start: i64, hashed: bool) {
        let mut a = start;
        loop {
            let mut b = self.nx(self.nx(a));
            while b != self.pv(a) {
                if self.vi(a) != self.vi(b) && self.valid_diagonal(a, b) {
                    let c = self.split_polygon(a, b);
                    let an = self.nx(a);
                    let a1 = self.filter_duplicates(a, an);
                    let cn = self.nx(c);
                    let c1 = self.filter_duplicates(c, cn);
                    self.earcut_linked(a1, hashed, 0);
                    self.earcut_linked(c1, hashed, 0);
                    return;
                }
                b = self.nx(b);
            }
            a = self.nx(a);
            if a == start {
                break;
            }
        }
        let s = self.filter(start, -1, true);
        self.earcut_linked(s, hashed, 3);
    }

    fn earcut_linked(&mut self, ear: i64, hashed: bool, stage: u8) {
        if ear < 0 {
            return;
        }
        if stage == 0 && hashed {
            self.index_curve(ear);
        }
        let mut ear = ear;
        let mut stop = ear;
        while self.pv(ear) != self.nx(ear) {
            let (prv, nxt) = (self.pv(ear), self.nx(ear));
            let is_ear = if hashed {
                self.is_ear_hashed(ear)
            } else {
                self.is_ear(ear)
            };
            if is_ear {
                let tri = [self.vi(prv), self.vi(ear), self.vi(nxt)];
                self.triangles.extend_from_slice(&tri);
                self.remove(ear);
                ear = self.nx(nxt);
                stop = ear;
                continue;
            }
            ear = nxt;
            if ear == stop {
                match stage {
                    0 => {
                        let e = self.filter_duplicates(ear, -1);
                        self.earcut_linked(e, hashed, 1);
                    }
                    1 => {
                        let e = self.filter_duplicates(ear, -1);
                        let e = self.cure_local_intersections(e);
                        self.earcut_linked(e, hashed, 2);
                    }
                    2 => self.split_earcut(ear, hashed),
                    _ => {}
                }
                break;
            }
        }
    }

    fn leftmost(&self, start: i64) -> i64 {
        let mut p = start;
        let mut best = start;
        loop {
            if self.px(p) < self.px(best)
                || (self.px(p) == self.px(best) && self.py(p) < self.py(best))
            {
                best = p;
            }
            p = self.nx(p);
            if p == start {
                return best;
            }
        }
    }

    fn sector(&self, m: i64, p: i64) -> bool {
        self.cross(self.pv(m), m, self.pv(p)) > 0.0 && self.cross(self.nx(p), m, self.nx(m)) > 0.0
    }

    fn find_bridge(&self, hole: i64, outer: i64) -> i64 {
        let mut p = outer;
        let (hx, hy) = (self.px(hole), self.py(hole));
        let mut qx = f64::NEG_INFINITY;
        let mut m = -1i64;
        loop {
            let q = self.nx(p);
            if hy <= self.py(p) && hy >= self.py(q) && self.py(q) != self.py(p) {
                let x = self.px(p)
                    + (hy - self.py(p)) * (self.px(q) - self.px(p)) / (self.py(q) - self.py(p));
                if x <= hx && x > qx {
                    qx = x;
                    m = if self.px(p) < self.px(q) { p } else { q };
                    if x == hx {
                        return m;
                    }
                }
            }
            p = q;
            if p == outer {
                break;
            }
        }
        if m < 0 {
            return -1;
        }
        let stop = m;
        let (mx, my) = (self.px(m), self.py(m));
        let mut tan_min = f64::INFINITY;
        let mut p = m;
        loop {
            let (px, py) = (self.px(p), self.py(p));
            if hx >= px && px >= mx && hx != px {
                let inside = if hy < my {
                    in_triangle(hx, hy, mx, my, qx, hy, px, py)
                } else {
                    in_triangle(qx, hy, mx, my, hx, hy, px, py)
                };
                if inside {
                    let tan = (hy - py).abs() / (hx - px);
                    if self.locally_inside(p, hole)
                        && (tan < tan_min
                            || (tan == tan_min
                                && (px > self.px(m) || (px == self.px(m) && self.sector(m, p)))))
                    {
                        m = p;
                        tan_min = tan;
                    }
                }
            }
            p = self.nx(p);
            if p == stop {
                return m;
            }
        }
    }

    fn run(&mut self, rings: &[(i64, i64)]) {
        let (r0, r1) = rings[0];
        if r1 - r0 < 3 {
            return;
        }
        let mut outer = self.ring(r0, r1, true);
        if outer < 0 || self.nx(outer) == self.pv(outer) {
            return;
        }
        let mut holes = Vec::new();
        for &(s, e) in &rings[1..] {
            if e - s < 3 {
                continue;
            }
            let h = self.ring(s, e, false);
            if h >= 0 && self.nx(h) != h {
                holes.push(self.leftmost(h));
            }
        }
        // Python sorts by (x, y) tuples: stable, lexicographic.
        holes.sort_by(|&a, &b| {
            let (xa, ya, xb, yb) = (self.px(a), self.py(a), self.px(b), self.py(b));
            xa.partial_cmp(&xb)
                .unwrap()
                .then(ya.partial_cmp(&yb).unwrap())
        });
        for h in holes {
            let bridge = self.find_bridge(h, outer);
            if bridge < 0 {
                continue;
            }
            let back = self.split_polygon(bridge, h);
            let bn = self.nx(back);
            self.filter_duplicates(back, bn);
            let brn = self.nx(bridge);
            outer = self.filter_duplicates(bridge, brn);
        }
        let mut count = 0usize;
        let mut p = outer;
        let (mut min_x, mut max_x) = (self.px(outer), self.px(outer));
        let (mut min_y, mut max_y) = (self.py(outer), self.py(outer));
        loop {
            count += 1;
            min_x = pmin(min_x, self.px(p));
            max_x = pmax(max_x, self.px(p));
            min_y = pmin(min_y, self.py(p));
            max_y = pmax(max_y, self.py(p));
            p = self.nx(p);
            if p == outer {
                break;
            }
        }
        let hashed = count > Z_ORDER_THRESHOLD;
        if hashed {
            self.min_x = min_x;
            self.min_y = min_y;
            let size = pmax(max_x - min_x, max_y - min_y);
            self.inv_size = if size != 0.0 { 32767.0 / size } else { 0.0 };
        }
        self.earcut_linked(outer, hashed, 0);
    }
}

/// Check the ring and polygon tables (the Python wrapper checks them too).
pub fn check(points: usize, rings: &[i64], polys: &[i64]) -> Result<usize, String> {
    if rings.is_empty() || rings[0] != 0 || rings[rings.len() - 1] != points as i64 {
        return Err("ring_starts must run from 0 to the point count".into());
    }
    if rings.windows(2).any(|w| w[1] < w[0]) {
        return Err("ring_starts must not decrease".into());
    }
    let nrings = rings.len() as i64 - 1;
    if polys.is_empty() || polys[0] != 0 || polys[polys.len() - 1] != nrings {
        return Err("polygon_starts must run from 0 to the ring count".into());
    }
    if polys.windows(2).any(|w| w[1] < w[0] + 1) {
        return Err("every polygon needs an outer ring".into());
    }
    // Room for the triangles: at most the points plus two per ring.
    Ok(points + 2 * nrings as usize)
}

/// Triangulate every polygon; writes `tris` (3 per triangle) and `owner`; returns the
/// triangle count, or None if the output buffers are too small.
pub fn triangulate(
    xy: &[f64],
    rings: &[i64],
    polys: &[i64],
    tris: &mut [i64],
    owner: &mut [i64],
) -> Option<usize> {
    let n = xy.len() / 2;
    let xs: Vec<f64> = (0..n).map(|k| xy[2 * k]).collect();
    let ys: Vec<f64> = (0..n).map(|k| xy[2 * k + 1]).collect();
    let mut count = 0usize;
    for j in 0..polys.len() - 1 {
        let (r0, r1) = (polys[j] as usize, polys[j + 1] as usize);
        let list: Vec<(i64, i64)> = (r0..r1).map(|k| (rings[k], rings[k + 1])).collect();
        let mut e = Earcut::new(&xs, &ys);
        e.run(&list);
        let t = e.triangles.len() / 3;
        if 3 * (count + t) > tris.len() || count + t > owner.len() {
            return None;
        }
        tris[3 * count..3 * (count + t)].copy_from_slice(&e.triangles);
        for o in owner[count..count + t].iter_mut() {
            *o = j as i64;
        }
        count += t;
    }
    Some(count)
}
