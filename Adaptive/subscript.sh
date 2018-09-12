#$ -P gpu
#$ -l gpu=1
#$ -l h_rt=23:59:0
#$ -l tmem=11.5G
#$ -S /bin/bash
#$ -o /home/talexopo/Adaptive/
#$ -e /home/talexopo/Adaptive/
#$ -wd /home/talexopo/Adaptive/NiftyNet/

source /share/apps/examples/python/python-3.6.5.source
source /share/apps/examples/cuda/cuda-9.0.source

# The following configuration is based on the example outlined in https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer/blob/master/mr_ct_regression_model_zoo.md

# In the absence of pre-defined error maps, Uniform Sampler is utilized for the first 735 iterations in the case of HH (or, equivalently, first 5 epochs) and inference 
# is carried out subsequently in order to deduce the first error maps.
LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.5-shared/bin/python3) -u net_regress.py train -c /home/talexopo/Adaptive/configuration_0.ini --starting_iter 0 --max_iter 735
LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.5-shared/bin/python3) -u net_run.py inference -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression -c /home/talexopo/Adaptive/configuration_1.ini --inference_iter -1 --spatial_window_size 96,96,96 --batch_size 4 --error_map True

# Then, every 1470 iterations in the case of HH (or, equivalently, 10 epochs) training is interrupted in order for inference to be carried out to update the respective error maps.
for max_iter in `seq 2205 1470 148470`
do
 LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.5-shared/bin/python3) -u net_run.py train -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression -c /home/talexopo/Adaptive/configuration_1.ini --starting_iter -1 --max_iter $max_iter
 LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.5-shared/bin/python3) -u net_run.py inference -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression -c /home/talexopo/Adaptive/configuration_1.ini --inference_iter -1 --spatial_window_size 96,96,96 --batch_size 4 --error_map True
done

LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.5-shared/bin/python3) -u net_regress.py inference -c /home/talexopo/Adaptive/configuration_1.ini