#$ -P gpu
#$ -l gpu=1
#$ -l h_rt=23:59:0
#$ -l tmem=11.5G
#$ -S /bin/bash
#$ -o /home/talexopo/VDSR/
#$ -e /home/talexopo/VDSR/
#$ -wd /home/talexopo/VDSR/NiftyNet/

source /share/apps/examples/python/python-3.6.5.source
source /share/apps/examples/cuda/cuda-9.0.source

LD_LIBRARY_PATH="/share/apps/python-3.6.5-shared/lib:${LD_LIBRARY_PATH}" $(command -v /share/apps/python-3.6.5-shared/bin/pip3) install --user -r requirements-gpu.txt

# Learning rate changes at a pre-determined number of epochs.

LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.5-shared/bin/python3) -u net_regress.py train -c /home/talexopo/VDSR/configuration_0.ini --lr 0.01 --starting_iter 0 --max_iter 1300

LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.5-shared/bin/python3) -u net_regress.py train -c /home/talexopo/VDSR/configuration_0.ini --lr 0.001 --starting_iter -1 --max_iter 1950

LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.5-shared/bin/python3) -u net_regress.py train -c /home/talexopo/VDSR/configuration_0.ini --lr 0.0001 --starting_iter -1 --max_iter 32500

LD_LIBRARY_PATH="/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.3-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}" /share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps/python-3.6.3-shared/bin/python3) -u net_regress.py inference -c /home/talexopo/VDSR/configuration_0.ini