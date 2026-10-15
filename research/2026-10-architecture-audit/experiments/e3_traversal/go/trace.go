// Scalar port of SceneAccel._trace (closest hit); idiomatic slices (bounds-checked), goroutines.
package main

/*
#include "trace.h"
*/
import "C"
import (
	"math"
	"sync"
	"sync/atomic"
	"unsafe"
)

const minDir = 1e-300
const stackMax = 256

type scene struct {
	child                  []int64
	childLo, childHi       []float32
	kind                   []int8
	first, count           []int64
	v0, v1, v2             []float32
	triPrim, slotInst      []int64
	instLo, instHi         []float64
	instRoot               []int64
	toObject, rootLo, rootHi []float64
}
type space struct {
	o, d, inv, shear [3]float64
	kx, ky, kz       int
}

func (s *space) setup() {
	for i := 0; i < 3; i++ {
		v := s.d[i]
		safe := v
		if math.Abs(v) < minDir {
			if v < 0 {
				safe = -minDir
			} else {
				safe = minDir
			}
		}
		s.inv[i] = 1.0 / safe
	}
	z := 0
	m := math.Abs(s.d[0])
	if math.Abs(s.d[1]) > m {
		z = 1
		m = math.Abs(s.d[1])
	}
	if math.Abs(s.d[2]) > m {
		z = 2
	}
	x := (z + 1) % 3
	y := (x + 1) % 3
	dz := s.d[z]
	if dz < 0 {
		x, y = y, x
	}
	dzs := dz
	if dz == 0 {
		dzs = 1
	}
	s.shear = [3]float64{s.d[x] / dzs, s.d[y] / dzs, 1.0 / dzs}
	s.kx, s.ky, s.kz = x, y, z
}
func mn(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
func mx(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
func (s *space) slab(lo, hi [3]float64, tmin, tmax float64) (bool, float64) {
	var a, b [3]float64
	for i := 0; i < 3; i++ {
		t1 := (lo[i] - s.o[i]) * s.inv[i]
		t2 := (hi[i] - s.o[i]) * s.inv[i]
		a[i] = mn(t1, t2)
		b[i] = mx(t1, t2)
	}
	near := mx(mx(a[0], a[1]), mx(a[2], tmin))
	far := mn(mn(b[0], b[1]), b[2]) * (1.0 + 4e-16)
	return near <= far && near < tmax, near
}
func f3(p []float32, k int) [3]float64 { return [3]float64{float64(p[k]), float64(p[k+1]), float64(p[k+2])} }
func d3(p []float64, k int) [3]float64 { return [3]float64{p[k], p[k+1], p[k+2]} }
func triangle(p0, p1, p2 [3]float64, s *space) (bool, float64, [3]float64) {
	a := [3]float64{p0[0] - s.o[0], p0[1] - s.o[1], p0[2] - s.o[2]}
	b := [3]float64{p1[0] - s.o[0], p1[1] - s.o[1], p1[2] - s.o[2]}
	c := [3]float64{p2[0] - s.o[0], p2[1] - s.o[1], p2[2] - s.o[2]}
	sx, sy, sz := s.shear[0], s.shear[1], s.shear[2]
	az, bz, cz := a[s.kz], b[s.kz], c[s.kz]
	ax := a[s.kx] - sx*az
	ay := a[s.ky] - sy*az
	bx := b[s.kx] - sx*bz
	by := b[s.ky] - sy*bz
	cx := c[s.kx] - sx*cz
	cy := c[s.ky] - sy*cz
	u := cx*by - cy*bx
	v := ax*cy - ay*cx
	w := bx*ay - by*ax
	det := u + v + w
	if ((u < 0 || v < 0 || w < 0) && (u > 0 || v > 0 || w > 0)) || det == 0 {
		return false, 0, [3]float64{}
	}
	t := (u*(sz*az) + v*(sz*bz) + w*(sz*cz)) / det
	return true, t, [3]float64{u / det, v / det, w / det}
}
func traceOne(sc *scene, ow, dw [3]float64, tmin, tmax float64, skI, skT int64, stack *[stackMax]int64, tstack *[stackMax]float64) (float64, int64, int64, [3]float64) {
	tBest := tmax
	hitTri, hitInst := int64(-1), int64(-1)
	var hb [3]float64
	finite := true
	for i := 0; i < 3; i++ {
		if math.IsInf(ow[i], 0) || math.IsNaN(ow[i]) || math.IsInf(dw[i], 0) || math.IsNaN(dw[i]) {
			finite = false
		}
	}
	s := space{o: ow, d: dw}
	s.setup()
	cur := int64(-1)
	sp := 0
	node := int64(0)
	ok, tNode := s.slab(d3(sc.rootLo, 0), d3(sc.rootHi, 0), tmin, tBest)
	active := finite && sc.rootLo[0] <= sc.rootHi[0] && ok
	for active {
		needPop := true
		if tNode < tBest {
			inst := (node >> 32) - 1
			nd := int(node & 0xFFFFFFFF)
			if inst != cur {
				if inst < 0 {
					s.o, s.d = ow, dw
				} else {
					m := sc.toObject[12*inst : 12*inst+12]
					for i := 0; i < 3; i++ {
						s.o[i] = m[4*i]*ow[0] + m[4*i+1]*ow[1] + m[4*i+2]*ow[2] + m[4*i+3]
						s.d[i] = m[4*i]*dw[0] + m[4*i+1]*dw[1] + m[4*i+2]*dw[2]
					}
				}
				s.setup()
				cur = inst
			}
			switch sc.kind[nd] {
			case 0:
				tag := (inst + 1) << 32
				ok0, tn0 := s.slab(f3(sc.childLo, 6*nd), f3(sc.childHi, 6*nd), tmin, tBest)
				ok1, tn1 := s.slab(f3(sc.childLo, 6*nd+3), f3(sc.childHi, 6*nd+3), tmin, tBest)
				if ok0 || ok1 {
					near0 := ok0
					if ok0 && ok1 {
						near0 = tn0 <= tn1
					}
					if near0 {
						node, tNode = tag|sc.child[2*nd], tn0
					} else {
						node, tNode = tag|sc.child[2*nd+1], tn1
					}
					if ok0 && ok1 {
						if near0 {
							stack[sp], tstack[sp] = tag|sc.child[2*nd+1], tn1
						} else {
							stack[sp], tstack[sp] = tag|sc.child[2*nd], tn0
						}
						sp++
					}
					needPop = false
				}
			case 2:
				f, c := int(sc.first[nd]), int(sc.count[nd])
				for j := 0; j < c; j++ {
					ins := int(sc.slotInst[f+j])
					if ok, tn := s.slab(d3(sc.instLo, 3*ins), d3(sc.instHi, 3*ins), tmin, tBest); ok {
						stack[sp], tstack[sp] = ((int64(ins)+1)<<32)|sc.instRoot[ins], tn
						sp++
					}
				}
			default:
				f, c := int(sc.first[nd]), int(sc.count[nd])
				for g := f; g < f+c; g++ {
					ok, t, b := triangle(f3(sc.v0, 3*g), f3(sc.v1, 3*g), f3(sc.v2, 3*g), &s)
					if !ok || !(t > tmin && t < tBest) {
						continue
					}
					if inst == skI && sc.triPrim[g] == skT {
						continue
					}
					tBest, hitTri, hitInst, hb = t, int64(g), inst, b
				}
			}
		}
		if needPop {
			if sp == 0 {
				break
			}
			sp--
			node, tNode = stack[sp], tstack[sp]
		}
	}
	if hitInst >= 0 {
		return tBest, hitInst, sc.triPrim[hitTri], hb
	}
	return math.Inf(1), -1, -1, [3]float64{}
}

var lengths [9]int

//export set_lengths
func set_lengths(l *C.longlong) {
	s := unsafe.Slice((*int64)(unsafe.Pointer(l)), 9)
	for i := range lengths {
		lengths[i] = int(s[i])
	}
}

func sl[T any](p unsafe.Pointer, n int) []T { return unsafe.Slice((*T)(p), n) }

//export closest_hit
func closest_hit(a *C.Accel, r *C.Rays, nthreads C.longlong) C.int {
	if a.stack_size > stackMax {
		return -1
	}
	L := lengths
	sc := scene{
		child: sl[int64](unsafe.Pointer(a.child), L[0]), childLo: sl[float32](unsafe.Pointer(a.child_lo), L[1]),
		childHi: sl[float32](unsafe.Pointer(a.child_hi), L[1]), kind: sl[int8](unsafe.Pointer(a.kind), L[2]),
		first: sl[int64](unsafe.Pointer(a.first), L[2]), count: sl[int64](unsafe.Pointer(a.count), L[2]),
		v0: sl[float32](unsafe.Pointer(a.v0), L[3]), v1: sl[float32](unsafe.Pointer(a.v1), L[3]), v2: sl[float32](unsafe.Pointer(a.v2), L[3]),
		triPrim: sl[int64](unsafe.Pointer(a.tri_prim), L[4]), slotInst: sl[int64](unsafe.Pointer(a.slot_inst), L[5]),
		instLo: sl[float64](unsafe.Pointer(a.inst_lo), L[6]), instHi: sl[float64](unsafe.Pointer(a.inst_hi), L[6]),
		instRoot: sl[int64](unsafe.Pointer(a.inst_root), L[7]), toObject: sl[float64](unsafe.Pointer(a.to_object), L[8]),
		rootLo: sl[float64](unsafe.Pointer(a.root_lo), 3), rootHi: sl[float64](unsafe.Pointer(a.root_hi), 3),
	}
	n := int(r.n)
	o, d := sl[float64](unsafe.Pointer(r.o), 3*n), sl[float64](unsafe.Pointer(r.d), 3*n)
	tmin, tmax := sl[float64](unsafe.Pointer(r.tmin), n), sl[float64](unsafe.Pointer(r.tmax), n)
	var ski, skt []int64
	if r.skip_inst != nil {
		ski = sl[int64](unsafe.Pointer(r.skip_inst), n)
		skt = sl[int64](unsafe.Pointer(r.skip_tri), n)
	}
	ot, oi, op, ob := sl[float64](unsafe.Pointer(r.t), n), sl[int64](unsafe.Pointer(r.inst), n), sl[int64](unsafe.Pointer(r.prim), n), sl[float64](unsafe.Pointer(r.b), 3*n)
	var next atomic.Int64
	work := func() {
		var stack [stackMax]int64
		var tstack [stackMax]float64
		for {
			s0 := int(next.Add(512)) - 512
			if s0 >= n {
				return
			}
			end := min(s0+512, n)
			for i := s0; i < end; i++ {
				skI, skT := int64(-2), int64(-2)
				if ski != nil {
					skI, skT = ski[i], skt[i]
				}
				t, inst, prim, b := traceOne(&sc, d3(o, 3*i), d3(d, 3*i), tmin[i], tmax[i], skI, skT, &stack, &tstack)
				ot[i], oi[i], op[i] = t, inst, prim
				ob[3*i], ob[3*i+1], ob[3*i+2] = b[0], b[1], b[2]
			}
		}
	}
	nt := max(int(nthreads), 1)
	if nt == 1 {
		work()
		return 0
	}
	var wg sync.WaitGroup
	for i := 0; i < nt; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); work() }()
	}
	wg.Wait()
	return 0
}

//export noop
func noop() {}

func main() {}
