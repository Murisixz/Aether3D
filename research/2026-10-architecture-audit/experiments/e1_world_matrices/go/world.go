package main

import "C"
import "unsafe"

//export world_matrices
func world_matrices(n C.longlong, parentP *C.longlong, posP, quatP, scaleP, outP *C.double) {
	cnt := int(n)
	parent := unsafe.Slice((*int64)(unsafe.Pointer(parentP)), cnt)
	pos := unsafe.Slice((*float64)(unsafe.Pointer(posP)), 3*cnt)
	quat := unsafe.Slice((*float64)(unsafe.Pointer(quatP)), 4*cnt)
	scale := unsafe.Slice((*float64)(unsafe.Pointer(scaleP)), 3*cnt)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outP)), 16*cnt)
	var m [16]float64
	for k := 0; k < cnt; k++ {
		x, y, z, w := quat[4*k], quat[4*k+1], quat[4*k+2], quat[4*k+3]
		s0, s1, s2 := scale[3*k], scale[3*k+1], scale[3*k+2]
		m = [16]float64{(1 - 2*(y*y+z*z)) * s0, (2 * (x*y - z*w)) * s1, (2 * (x*z + y*w)) * s2, pos[3*k],
			(2 * (x*y + z*w)) * s0, (1 - 2*(x*x+z*z)) * s1, (2 * (y*z - x*w)) * s2, pos[3*k+1],
			(2 * (x*z - y*w)) * s0, (2 * (y*z + x*w)) * s1, (1 - 2*(x*x+y*y)) * s2, pos[3*k+2],
			0, 0, 0, 1}
		o := out[16*k : 16*k+16]
		p := parent[k]
		if p < 0 {
			copy(o, m[:])
			continue
		}
		a := out[16*int(p) : 16*int(p)+16]
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				o[4*i+j] = a[4*i]*m[j] + a[4*i+1]*m[4+j] + a[4*i+2]*m[8+j] + a[4*i+3]*m[12+j]
			}
		}
	}
}

//export noop
func noop() {}

func main() {}
