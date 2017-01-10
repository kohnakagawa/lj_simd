TARGET= aos.out aos_pair.out aos_intrin.out soa.out soa_pair.out soa_intrin.out aos_intrin_mat_transpose.out aos_brute_force1x4.out aos_brute_force4x1.out aos_brute_force_ref.out aos_brute_force1x4_recless.out aos_brute_force4x1_recless.out aos_brute_force_ref_rectless.out

ASM = force_aos.s force_soa.s force_aos_brute.s

all: $(TARGET) $(ASM)

.SUFFIXES:
.SUFFIXES: .cpp .s
.cpp.s:
	icpc -O3 -xHOST -std=c++11 -S -masm=intel $< -o $@

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

aos_brute_force1x4.out: force_aos_brute.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE1x4 $< -o $@

aos_brute_force4x1.out: force_aos_brute.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE4x1 $< -o $@

aos_brute_force_ref.out: force_aos_brute.cpp
	icpc -O3 -xHOST -std=c++11 -DREFERENCE $< -o $@

aos_brute_force1x4_recless.out: force_aos_brute.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE1x4_REACTLESS $< -o $@

aos_brute_force4x1_recless.out: force_aos_brute.cpp
	icpc -O3 -xHOST -std=c++11 -DUSE4x1_REACTLESS $< -o $@

aos_brute_force_ref_rectless.out: force_aos_brute.cpp
	icpc -O3 -xHOST -std=c++11 -DREFERENCE_REACTLESS $< -o $@

clean:
	rm -f $(TARGET)

test: aos_pair.out aos_intrin.out soa_pair.out soa_intrin.out aos_intrin_mat_transpose.out
	./aos_pair.out > aos_pair.dat
	./aos_intrin.out > aos_intrin.dat
	./aos_intrin_mat_transpose.out > aos_intrin_mat_transpose.dat
	diff aos_pair.dat aos_intrin.dat
	diff aos_intrin.dat aos_intrin_mat_transpose.dat
	./soa_pair.out > soa_pair.dat
	./soa_intrin.out > soa_intrin.dat
	diff soa_pair.dat soa_intrin.dat
