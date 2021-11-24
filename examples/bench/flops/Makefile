all:
	python -m turingas.main -i fp32ffma.sass -o fp32ffma.cubin
	nvcc -O0 -Xcicc -O0 -Xptxas -O0  -arch=sm_75 peak.cu -lcuda -o test

clean:
	rm fp32ffma.cubin test
