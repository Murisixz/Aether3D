# MikkTSpace reference tangents

`reference.npz` holds, for each case, triangle corner arrays (`<case>_positions`,
`<case>_normals`, `<case>_uvs`, float32) and the tangents (`<case>_tangents`: xyz and
sign per corner) computed by Morten S. Mikkelsen's reference implementation
(`mikktspace.c` from https://github.com/mmikk/MikkTSpace, zlib license), compiled with
`gcc -O2 -ffp-contract=off` on x86-64 and called through `genTangSpaceDefault` with
`setTSpaceBasic`. Aether3D's port (`aether3d/uv/mikktspace.py`) must reproduce them bit
for bit (`tests/test_tangents.py`). The C code is not part of Aether3D.

Cases: a 12x6 UV sphere, the unit cube, a noisy 8x8 grid with random normals, the same
with half of its UVs mirrored, with degenerate UV triangles, with degenerate position
triangles, and with duplicated (non-manifold) triangles. `branch_*` is a fan of eight
triangles around a vertex whose UVs are `z^2` of the positions: opposite triangles have
exactly opposite UV derivatives, so the vertex's group splits into subgroups.

`quirk_*` is the noisy grid with every ninth triangle duplicated. There the reference
pairs a non-manifold edge differently from the port: its edge sort (`BuildNeighborsFast`)
sorts each group of equal first indices by second index, and each group of equal edges by
triangle, *except the last group of the array*, which it leaves in quicksort order. The
port pairs every group in sorted order. Only which triangles of a non-manifold edge (or of
a last group with more than two entries) become neighbours can differ; on manifold meshes
the results are identical. `tests/test_tangents.py` checks that the difference is confined
to the three triangles around that edge.
