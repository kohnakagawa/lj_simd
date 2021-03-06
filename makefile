TARGET= aos.out aos_pair.out aos_intrin.out soa.out soa_pair.out soa_intrin.out aos_intrin_mat_transpose.out comp_1x4.out comp_4x1.out comp_ref_4.out comp_8x1_v1.out comp_8x1_v2.out comp_1x8_v1.out comp_1x8_v2.out comp_ref_8.out

ASM = force_aos.s force_soa.s comp_4x1_1x4.s comp_8x1_1x8.s

all: $(TARGET) $(ASM)

.SUFFIXES:
.SUFFIXES: .cpp .s
.cpp.s:
	icpc -O3 -xHOST -std=c++11 -S $< -o $@

aos.out: force_aos.cpp
	icpc -O3 -xHOST -std=c++11 $< -o $@

aos_pair.out: force_aos.cpp
	icpc -O3 -xHOST -std=c++11 -DPAIR $< -o $@

aos_intrin.out: force_aos.cpp
	icpc -O3 -xHOST -std=c++11 -DINTRIN $< -o $@

aos_intrin_mat_transpose.out: force_aos.cpp
	icpc -O3 -xHOST -std=c++11 -DMAT_TRANSPOSE $< -o $@

soa.out: force_soa.cpp
	icpc -O3 -xHOST -std=c++11 $< -o $@

soa_pair.out: force_soa.cpp
	icpc -O3 -xHOST -std=c++11 -DPAIR $< -o $@

soa_intrin.out: force_soa.cpp
	icpc -O3 -xHOST -std=c++11 -DINTRIN $< -o $@

comp_1x4.out: comp_4x1_1x4.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE1x4 $< -o $@

comp_4x1.out: comp_4x1_1x4.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE4x1 $< -o $@

comp_ref_4.out: comp_4x1_1x4.cpp
	icpc -O3 -xHOST -std=c++11 -DREFERENCE $< -o $@

comp_1x8_v1.out: comp_8x1_1x8.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE1x8_v1 $< -o $@

comp_1x8_v2.out: comp_8x1_1x8.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE1x8_v2 $< -o $@

comp_8x1_v1.out: comp_8x1_1x8.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE8x1_v1 $< -o $@

comp_8x1_v2.out: comp_8x1_1x8.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE8x1_v2 $< -o $@

comp_ref_8.out: comp_8x1_1x8.cpp
	icpc -O3 -xHOST -std=c++11 -DREFERENCE $< -o $@

clean:
	rm -f $(TARGET) $(ASM)

test: aos_pair.out aos_intrin.out soa_pair.out soa_intrin.out aos_intrin_mat_transpose.out
	./aos_pair.out > aos_pair.dat
	./aos_intrin.out > aos_intrin.dat
	./aos_intrin_mat_transpose.out > aos_intrin_mat_transpose.dat
	diff aos_pair.dat aos_intrin.dat
	diff aos_intrin.dat aos_intrin_mat_transpose.dat
	./soa_pair.out > soa_pair.dat
	./soa_intrin.out > soa_intrin.dat
	diff soa_pair.dat soa_intrin.dat
