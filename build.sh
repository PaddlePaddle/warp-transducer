
# https://colab.research.google.com/drive/1vMkH8LmiCCOiCo4KTTEcv-NU8_OGn0ie?usp=sharing#scrollTo=DWEKaNrsV1mD
 git clone --single-branch --branch espnet_v1.1 https://github.com/b-flo/warp-transducer.git
 cd warp-transducer && \
 mkdir build && \
 cd build && \
 cmake -DCMAKE_BUILD_TYPE=Release .. && \
 make -j
