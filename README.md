# AMLS_II_assignment20_21
Research project for AMLS II (ELEC0135)


The project script details the parameters for training the agent and for testing it against the semi-random opponent:
https://github.com/uceerbr/AMLS_II_assignment20_21/blob/33e6e65b5955f0fb3280eb307077a4fc52cb23ae/connect4_submission.ipynb
Just for fun, I have included this code the user would like to play either the semi-random opponent or the trained agent:
https://github.com/uceerbr/AMLS_II_assignment20_21/blob/main/connect4_PlayTheAgent.ipynb

Either piece of code requires access to the DQN model in the directory ./Trainings_500Ktraining/models4/dqn_training_rewardEveryMove

I ran the program in an environment that had these libraries installed:


absl-py                   0.9.0            py36h9f0ad1d_1    conda-forge/label/cf202003

argon2-cffi               21.1.0           py36h68aa20f_0    conda-forge

astor                     0.7.1                      py_0    conda-forge/label/cf202003

async_generator           1.10                       py_0    conda-forge

attrs                     21.4.0             pyhd8ed1ab_0    conda-forge

backcall                  0.2.0              pyh9f0ad1d_0    conda-forge

backports                 1.0                        py_2    conda-forge

backports.functools_lru_cache 1.6.4              pyhd8ed1ab_0    conda-forge

bleach                    4.1.0              pyhd8ed1ab_0    conda-forge

ca-certificates           2021.10.8            h5b45459_0    conda-forge

certifi                   2021.5.30        py36ha15d459_0    conda-forge

cffi                      1.14.6           py36he58ceb7_1    conda-forge

cmake                     3.22.2                   pypi_0    pypi

colorama                  0.4.4              pyh9f0ad1d_0    conda-forge

console_shortcut          0.1.1                         4

cycler                    0.11.0             pyhd8ed1ab_0    conda-forge

decorator                 5.1.1              pyhd8ed1ab_0    conda-forge

defusedxml                0.7.1              pyhd8ed1ab_0    conda-forge

dlib                      19.22.0          py36h9f2b841_0    conda-forge

entrypoints               0.3             pyhd8ed1ab_1003    conda-forge

freetype                  2.10.4               h546665d_1    conda-forge

gast                      0.3.3                      py_0    conda-forge/label/cf202003

graphviz                  2.38                 hfd603c8_2

grpcio                    1.14.1           py36he089993_0    conda-forge/label/cf202003

h5py                      2.10.0          nompi_py36h422b98e_102    conda-forge/label/cf202003

hdf5                      1.10.5          nompi_ha405e13_1104    conda-forge/label/cf202003

icu                       58.2                 ha925a31_3

importlib-metadata        4.8.1            py36ha15d459_0    conda-forge

intel-openmp              2022.0.0          h57928b3_3663    conda-forge

ipykernel                 5.5.5            py36hfacbf0b_0    conda-forge

ipython                   7.16.1           py36h7b2dad6_2    conda-forge

ipython_genutils          0.2.0                      py_1    conda-forge

jedi                      0.17.2           py36ha15d459_1    conda-forge

jinja2                    3.0.3              pyhd8ed1ab_0    conda-forge

jpeg                      9e                   h8ffe710_0    conda-forge

jsonschema                4.1.2              pyhd8ed1ab_0    conda-forge

jupyter_client            7.1.2              pyhd8ed1ab_0    conda-forge

jupyter_core              4.8.1            py36ha15d459_0    conda-forge

jupyterlab_pygments       0.1.2              pyh9f0ad1d_0    conda-forge

kaggle-environments       1.8.12                   pypi_0    pypi

keras                     2.3.1            py36h21ff451_0    conda-forge

keras-applications        1.0.8                      py_1    conda-forge/label/cf202003

keras-preprocessing       1.1.0                      py_0    conda-forge/label/cf202003

kiwisolver                1.3.1            py36he95197e_1    conda-forge

lcms2                     2.12                 h2a16943_0    conda-forge

libblas                   3.8.0                    14_mkl    conda-forge/label/cf202003

libcblas                  3.8.0                    14_mkl    conda-forge/label/cf202003

libgpuarray               0.7.6             h8ffe710_1003    conda-forge

liblapack                 3.8.0                    14_mkl    conda-forge/label/cf202003

liblapacke                3.8.0                    14_mkl    conda-forge

libpng                    1.6.37               h1d00b33_2    conda-forge

libprotobuf               3.11.4               h1a1b453_0    conda-forge/label/cf202003

libsodium                 1.0.18               h8d14728_1    conda-forge

libtiff                   4.2.0                h0c97f57_3    conda-forge

libwebp                   1.0.2                hfa6e2cd_5    conda-forge

lz4-c                     1.9.3                h8ffe710_1    conda-forge

m2w64-gcc-libgfortran     5.3.0                         6    conda-forge

m2w64-gcc-libs            5.3.0                         7    conda-forge

m2w64-gcc-libs-core       5.3.0                         7    conda-forge

m2w64-gmp                 6.1.0                         2    conda-forge

m2w64-libwinpthread-git   5.0.0.4634.697f757               2    conda-forge

mako                      1.1.6              pyhd8ed1ab_0    conda-forge

markdown                  3.2.1                      py_0    conda-forge/label/cf202003

markupsafe                2.0.1            py36h68aa20f_0    conda-forge

matplotlib                3.3.4            py36ha15d459_0    conda-forge

matplotlib-base           3.3.4            py36h1abdf75_0    conda-forge

mistune                   0.8.4           py36h68aa20f_1004    conda-forge

mkl                       2019.4                      245

msys2-conda-epoch         20160418                      1    conda-forge

nbclient                  0.5.9              pyhd8ed1ab_0    conda-forge

nbconvert                 6.0.7            py36ha15d459_3    conda-forge

nbformat                  5.1.3              pyhd8ed1ab_0    conda-forge

nest-asyncio              1.5.4              pyhd8ed1ab_0    conda-forge

notebook                  6.3.0            py36ha15d459_0    conda-forge

numpy                     1.18.1           py36hc71023c_0    conda-forge/label/cf202003

olefile                   0.46               pyh9f0ad1d_1    conda-forge

opencv                    4.1.0            py36hce2de41_1    conda-forge

openjpeg                  2.4.0                hb211442_1    conda-forge

openssl                   1.0.2u               hfa6e2cd_0    conda-forge

packaging                 21.3               pyhd8ed1ab_0    conda-forge

pandas                    0.25.3           py36he350917_0    conda-forge

pandoc                    2.17.0.1             h57928b3_1    conda-forge

pandocfilters             1.5.0              pyhd8ed1ab_0    conda-forge

parso                     0.7.1              pyh9f0ad1d_0    conda-forge

patsy                     0.5.2              pyhd8ed1ab_0    conda-forge

pickleshare               0.7.5                   py_1003    conda-forge

pillow                    8.2.0            py36h9cbe6be_1    conda-forge

pip                       21.3.1             pyhd8ed1ab_0    conda-forge

prometheus_client         0.13.1             pyhd8ed1ab_0    conda-forge

prompt-toolkit            3.0.26             pyha770c72_0    conda-forge

protobuf                  3.11.4           py36he025d50_0    conda-forge/label/cf202003

pycparser                 2.21               pyhd8ed1ab_0    conda-forge

pydot                     1.4.2            py36ha15d459_0    conda-forge

pygments                  2.11.2             pyhd8ed1ab_0    conda-forge

pygpu                     0.7.6           py36h6434af4_1002    conda-forge

pyparsing                 3.0.7              pyhd8ed1ab_0    conda-forge

pyqt                      5.6.0           py36h764d66f_1008    conda-forge

pyreadline                2.1                   py36_1001    conda-forge/label/cf202003

pyrsistent                0.17.3           py36h68aa20f_2    conda-forge

python                    3.6.15          h39d44d4_0_cpython    conda-forge

python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge

python_abi                3.6                     2_cp36m    conda-forge

pytz                      2021.3             pyhd8ed1ab_0    conda-forge

pywin32                   301              py36h68aa20f_0    conda-forge

pywinpty                  1.1.4            py36hcae0e51_0    conda-forge

pyyaml                    5.4.1            py36h68aa20f_1    conda-forge

pyzmq                     22.3.0           py36h1d5d788_0    conda-forge

qt                        5.6.2           vc14h6f8c307_12

scipy                     1.5.3            py36h7ff6e69_0    conda-forge

seaborn                   0.11.2               hd8ed1ab_0    conda-forge

seaborn-base              0.11.2             pyhd8ed1ab_0    conda-forge

send2trash                1.8.0              pyhd8ed1ab_0    conda-forge

setuptools                49.6.0           py36ha15d459_3    conda-forge

sip                       4.18.1          py36h6538335_1000    conda-forge

six                       1.16.0             pyh6c4a22f_0    conda-forge

sqlite                    3.37.0               h8ffe710_0    conda-forge

statsmodels               0.12.1           py36h6434af4_2    conda-forge

tensorboard               1.13.1                   py36_0    conda-forge/label/cf202003

tensorflow                1.13.2               h21ff451_0    conda-forge/label/cf202003

tensorflow-base           1.13.2                   py36_0    conda-forge/label/cf202003

tensorflow-estimator      1.13.0           py36h39e3cac_0    conda-forge/label/cf202003

termcolor                 1.1.0                      py_2    conda-forge/label/cf202003

terminado                 0.12.1           py36ha15d459_0    conda-forge

testpath                  0.5.0              pyhd8ed1ab_0    conda-forge

theano                    1.0.5            py36he2d232f_1    conda-forge

tk                        8.6.11               h8ffe710_1    conda-forge

tornado                   6.1              py36h68aa20f_1    conda-forge

traitlets                 4.3.3              pyhd8ed1ab_2    conda-forge

typing_extensions         4.0.1              pyha770c72_0    conda-forge

ucrt                      10.0.20348.0         h57928b3_0    conda-forge

vc                        14.2                 hb210afc_6    conda-forge

vs2015_runtime            14.29.30037          h902a5da_6    conda-forge

vs2017_win-64             19.16.27033          hb90652a_6    conda-forge

vswhere                   2.8.4                h57928b3_0    conda-forge

wcwidth                   0.2.5              pyh9f0ad1d_2    conda-forge

webencodings              0.5.1                      py_1    conda-forge

werkzeug                  1.0.0                      py_0    conda-forge/label/cf202003

wheel                     0.37.1             pyhd8ed1ab_0    conda-forge

wincertstore              0.2             py36ha15d459_1006    conda-forge

winpty                    0.4.3                         4    conda-forge

xz                        5.2.5                h62dcd97_1    conda-forge

yaml                      0.2.5                h8ffe710_2    conda-forge

zeromq                    4.3.4                h0e60522_1    conda-forge

zipp                      3.7.0              pyhd8ed1ab_0    conda-forge

zlib                      1.2.11                   vc14_0  [vc14]  conda-forge/label/cf202003

zstd                      1.5.0                h6255e5f_0    conda-forge
