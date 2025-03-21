��!
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/v/dense/kernel
{
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/m/dense/kernel
{
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes

:*
dtype0
|
Adam/v/conv_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv_7/bias
u
&Adam/v/conv_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv_7/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv_7/bias
u
&Adam/m/conv_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv_7/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*%
shared_nameAdam/v/conv_7/kernel
�
(Adam/v/conv_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv_7/kernel*&
_output_shapes
:A*
dtype0
�
Adam/m/conv_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*%
shared_nameAdam/m/conv_7/kernel
�
(Adam/m/conv_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv_7/kernel*&
_output_shapes
:A*
dtype0
�
Adam/v/batchnorm_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/batchnorm_6/beta

+Adam/v/batchnorm_6/beta/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_6/beta*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/batchnorm_6/beta

+Adam/m/batchnorm_6/beta/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_6/beta*
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/batchnorm_6/gamma
�
,Adam/v/batchnorm_6/gamma/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_6/gamma*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/batchnorm_6/gamma
�
,Adam/m/batchnorm_6/gamma/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_6/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv_6/bias
u
&Adam/v/conv_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv_6/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv_6/bias
u
&Adam/m/conv_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv_6/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv_6/kernel
�
(Adam/v/conv_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv_6/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv_6/kernel
�
(Adam/m/conv_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv_6/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/batchnorm_5/beta

+Adam/v/batchnorm_5/beta/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_5/beta*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/batchnorm_5/beta

+Adam/m/batchnorm_5/beta/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_5/beta*
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/batchnorm_5/gamma
�
,Adam/v/batchnorm_5/gamma/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_5/gamma*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/batchnorm_5/gamma
�
,Adam/m/batchnorm_5/gamma/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_5/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv_5/bias
u
&Adam/v/conv_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv_5/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv_5/bias
u
&Adam/m/conv_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv_5/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv_5/kernel
�
(Adam/v/conv_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv_5/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv_5/kernel
�
(Adam/m/conv_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv_5/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/batchnorm_4/beta

+Adam/v/batchnorm_4/beta/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_4/beta*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/batchnorm_4/beta

+Adam/m/batchnorm_4/beta/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_4/beta*
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/batchnorm_4/gamma
�
,Adam/v/batchnorm_4/gamma/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_4/gamma*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/batchnorm_4/gamma
�
,Adam/m/batchnorm_4/gamma/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_4/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv_4/bias
u
&Adam/v/conv_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv_4/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv_4/bias
u
&Adam/m/conv_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv_4/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv_4/kernel
�
(Adam/v/conv_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv_4/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv_4/kernel
�
(Adam/m/conv_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv_4/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/batchnorm_3/beta

+Adam/v/batchnorm_3/beta/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_3/beta*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/batchnorm_3/beta

+Adam/m/batchnorm_3/beta/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_3/beta*
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/batchnorm_3/gamma
�
,Adam/v/batchnorm_3/gamma/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_3/gamma*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/batchnorm_3/gamma
�
,Adam/m/batchnorm_3/gamma/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_3/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv_3/bias
u
&Adam/v/conv_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv_3/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv_3/bias
u
&Adam/m/conv_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv_3/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv_3/kernel
�
(Adam/v/conv_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv_3/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv_3/kernel
�
(Adam/m/conv_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv_3/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/batchnorm_2/beta

+Adam/v/batchnorm_2/beta/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_2/beta*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/batchnorm_2/beta

+Adam/m/batchnorm_2/beta/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_2/beta*
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/batchnorm_2/gamma
�
,Adam/v/batchnorm_2/gamma/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_2/gamma*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/batchnorm_2/gamma
�
,Adam/m/batchnorm_2/gamma/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_2/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv_2/bias
u
&Adam/v/conv_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv_2/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv_2/bias
u
&Adam/m/conv_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv_2/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv_2/kernel
�
(Adam/v/conv_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv_2/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv_2/kernel
�
(Adam/m/conv_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv_2/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/batchnorm_1/beta

+Adam/v/batchnorm_1/beta/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_1/beta*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/batchnorm_1/beta

+Adam/m/batchnorm_1/beta/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_1/beta*
_output_shapes
:*
dtype0
�
Adam/v/batchnorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/batchnorm_1/gamma
�
,Adam/v/batchnorm_1/gamma/Read/ReadVariableOpReadVariableOpAdam/v/batchnorm_1/gamma*
_output_shapes
:*
dtype0
�
Adam/m/batchnorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/batchnorm_1/gamma
�
,Adam/m/batchnorm_1/gamma/Read/ReadVariableOpReadVariableOpAdam/m/batchnorm_1/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv_1/bias
u
&Adam/v/conv_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv_1/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv_1/bias
u
&Adam/m/conv_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv_1/kernel
�
(Adam/v/conv_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv_1/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv_1/kernel
�
(Adam/m/conv_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv_1/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
n
conv_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_7/bias
g
conv_7/bias/Read/ReadVariableOpReadVariableOpconv_7/bias*
_output_shapes
:*
dtype0
~
conv_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_nameconv_7/kernel
w
!conv_7/kernel/Read/ReadVariableOpReadVariableOpconv_7/kernel*&
_output_shapes
:A*
dtype0
�
batchnorm_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_6/moving_variance
�
/batchnorm_6/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_6/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_6/moving_mean

+batchnorm_6/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_6/moving_mean*
_output_shapes
:*
dtype0
x
batchnorm_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_6/beta
q
$batchnorm_6/beta/Read/ReadVariableOpReadVariableOpbatchnorm_6/beta*
_output_shapes
:*
dtype0
z
batchnorm_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_6/gamma
s
%batchnorm_6/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_6/gamma*
_output_shapes
:*
dtype0
n
conv_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_6/bias
g
conv_6/bias/Read/ReadVariableOpReadVariableOpconv_6/bias*
_output_shapes
:*
dtype0
~
conv_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_6/kernel
w
!conv_6/kernel/Read/ReadVariableOpReadVariableOpconv_6/kernel*&
_output_shapes
:*
dtype0
�
batchnorm_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_5/moving_variance
�
/batchnorm_5/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_5/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_5/moving_mean

+batchnorm_5/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_5/moving_mean*
_output_shapes
:*
dtype0
x
batchnorm_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_5/beta
q
$batchnorm_5/beta/Read/ReadVariableOpReadVariableOpbatchnorm_5/beta*
_output_shapes
:*
dtype0
z
batchnorm_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_5/gamma
s
%batchnorm_5/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_5/gamma*
_output_shapes
:*
dtype0
n
conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_5/bias
g
conv_5/bias/Read/ReadVariableOpReadVariableOpconv_5/bias*
_output_shapes
:*
dtype0
~
conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_5/kernel
w
!conv_5/kernel/Read/ReadVariableOpReadVariableOpconv_5/kernel*&
_output_shapes
:*
dtype0
�
batchnorm_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_4/moving_variance
�
/batchnorm_4/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_4/moving_mean

+batchnorm_4/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_mean*
_output_shapes
:*
dtype0
x
batchnorm_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_4/beta
q
$batchnorm_4/beta/Read/ReadVariableOpReadVariableOpbatchnorm_4/beta*
_output_shapes
:*
dtype0
z
batchnorm_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_4/gamma
s
%batchnorm_4/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_4/gamma*
_output_shapes
:*
dtype0
n
conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/bias
g
conv_4/bias/Read/ReadVariableOpReadVariableOpconv_4/bias*
_output_shapes
:*
dtype0
~
conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/kernel
w
!conv_4/kernel/Read/ReadVariableOpReadVariableOpconv_4/kernel*&
_output_shapes
:*
dtype0
�
batchnorm_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_3/moving_variance
�
/batchnorm_3/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_3/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_3/moving_mean

+batchnorm_3/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_3/moving_mean*
_output_shapes
:*
dtype0
x
batchnorm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_3/beta
q
$batchnorm_3/beta/Read/ReadVariableOpReadVariableOpbatchnorm_3/beta*
_output_shapes
:*
dtype0
z
batchnorm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_3/gamma
s
%batchnorm_3/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_3/gamma*
_output_shapes
:*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:*
dtype0
~
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/kernel
w
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*&
_output_shapes
:*
dtype0
�
batchnorm_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_2/moving_variance
�
/batchnorm_2/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_2/moving_mean

+batchnorm_2/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_mean*
_output_shapes
:*
dtype0
x
batchnorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_2/beta
q
$batchnorm_2/beta/Read/ReadVariableOpReadVariableOpbatchnorm_2/beta*
_output_shapes
:*
dtype0
z
batchnorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_2/gamma
s
%batchnorm_2/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_2/gamma*
_output_shapes
:*
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
:*
dtype0
~
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/kernel
w
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*&
_output_shapes
:*
dtype0
�
batchnorm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_1/moving_variance
�
/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_1/moving_mean

+batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_mean*
_output_shapes
:*
dtype0
x
batchnorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_1/beta
q
$batchnorm_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_1/beta*
_output_shapes
:*
dtype0
z
batchnorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_1/gamma
s
%batchnorm_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_1/gamma*
_output_shapes
:*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:*
dtype0
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
:*
dtype0
�
serving_default_conv_1_inputPlaceholder*/
_output_shapes
:���������A*
dtype0*$
shape:���������A
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv_1_inputconv_1/kernelconv_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv_2/kernelconv_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv_3/kernelconv_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv_4/kernelconv_4/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv_5/kernelconv_5/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv_6/kernelconv_6/biasbatchnorm_6/gammabatchnorm_6/betabatchnorm_6/moving_meanbatchnorm_6/moving_varianceconv_7/kernelconv_7/biasdense/kernel
dense/bias*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*2
config_proto" 

CPU

GPU2 *0J 8� *0
f+R)
'__inference_signature_wrapper_710839215

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
test_xspectra
test_features
test_segments
test_audio_list
 dirs

!params
"params_feat
#	optimizer
$
signatures*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op*
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4axis
	5gamma
6beta
7moving_mean
8moving_variance*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses* 
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
 G_jit_compiled_convolution_op*
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

_kernel
`bias
 a_jit_compiled_convolution_op*
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance*
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias
 {_jit_compiled_convolution_op*
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
+0
,1
52
63
74
85
E6
F7
O8
P9
Q10
R11
_12
`13
i14
j15
k16
l17
y18
z19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39*
�
+0
,1
52
63
E4
F5
O6
P7
_8
`9
i10
j11
y12
z13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27*
v
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

+0
,1*

+0
,1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
50
61
72
83*

50
61*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

E0
F1*

E0
F1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
O0
P1
Q2
R3*

O0
P1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

_0
`1*

_0
`1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
i0
j1
k2
l3*

i0
j1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

y0
z1*

y0
z1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEbatchnorm_5/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_5/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_5/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_5/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEconv_6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv_6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
a[
VARIABLE_VALUEbatchnorm_6/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatchnorm_6/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEbatchnorm_6/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEbatchnorm_6/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEconv_7/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv_7/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
`
70
81
Q2
R3
k4
l5
�6
�7
�8
�9
�10
�11*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19*

�0*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27*
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15
�trace_16
�trace_17
�trace_18
�trace_19
�trace_20
�trace_21
�trace_22
�trace_23
�trace_24
�trace_25
�trace_26
�trace_27* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 

70
81*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 

Q0
R1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 

k0
l1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
_Y
VARIABLE_VALUEAdam/m/conv_1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv_1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv_1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv_1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/batchnorm_1/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/batchnorm_1/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/batchnorm_1/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/batchnorm_1/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/batchnorm_2/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/batchnorm_2/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/batchnorm_2/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/batchnorm_2/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv_3/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv_3/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv_3/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv_3/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/batchnorm_3/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/batchnorm_3/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/batchnorm_3/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/batchnorm_3/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv_4/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv_4/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv_4/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv_4/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/batchnorm_4/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/batchnorm_4/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/batchnorm_4/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/batchnorm_4/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv_5/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv_5/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv_5/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv_5/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/batchnorm_5/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/batchnorm_5/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/batchnorm_5/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/batchnorm_5/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv_6/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv_6/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv_6/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv_6/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/batchnorm_6/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/batchnorm_6/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/batchnorm_6/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/batchnorm_6/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv_7/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv_7/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv_7/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv_7/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv_2/kernelconv_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv_3/kernelconv_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv_4/kernelconv_4/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv_5/kernelconv_5/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv_6/kernelconv_6/biasbatchnorm_6/gammabatchnorm_6/betabatchnorm_6/moving_meanbatchnorm_6/moving_varianceconv_7/kernelconv_7/biasdense/kernel
dense/bias	iterationlearning_rateAdam/m/conv_1/kernelAdam/v/conv_1/kernelAdam/m/conv_1/biasAdam/v/conv_1/biasAdam/m/batchnorm_1/gammaAdam/v/batchnorm_1/gammaAdam/m/batchnorm_1/betaAdam/v/batchnorm_1/betaAdam/m/conv_2/kernelAdam/v/conv_2/kernelAdam/m/conv_2/biasAdam/v/conv_2/biasAdam/m/batchnorm_2/gammaAdam/v/batchnorm_2/gammaAdam/m/batchnorm_2/betaAdam/v/batchnorm_2/betaAdam/m/conv_3/kernelAdam/v/conv_3/kernelAdam/m/conv_3/biasAdam/v/conv_3/biasAdam/m/batchnorm_3/gammaAdam/v/batchnorm_3/gammaAdam/m/batchnorm_3/betaAdam/v/batchnorm_3/betaAdam/m/conv_4/kernelAdam/v/conv_4/kernelAdam/m/conv_4/biasAdam/v/conv_4/biasAdam/m/batchnorm_4/gammaAdam/v/batchnorm_4/gammaAdam/m/batchnorm_4/betaAdam/v/batchnorm_4/betaAdam/m/conv_5/kernelAdam/v/conv_5/kernelAdam/m/conv_5/biasAdam/v/conv_5/biasAdam/m/batchnorm_5/gammaAdam/v/batchnorm_5/gammaAdam/m/batchnorm_5/betaAdam/v/batchnorm_5/betaAdam/m/conv_6/kernelAdam/v/conv_6/kernelAdam/m/conv_6/biasAdam/v/conv_6/biasAdam/m/batchnorm_6/gammaAdam/v/batchnorm_6/gammaAdam/m/batchnorm_6/betaAdam/v/batchnorm_6/betaAdam/m/conv_7/kernelAdam/v/conv_7/kernelAdam/m/conv_7/biasAdam/v/conv_7/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biastotalcountConst*q
Tinj
h2f*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_save_710840666
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv_2/kernelconv_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv_3/kernelconv_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv_4/kernelconv_4/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv_5/kernelconv_5/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv_6/kernelconv_6/biasbatchnorm_6/gammabatchnorm_6/betabatchnorm_6/moving_meanbatchnorm_6/moving_varianceconv_7/kernelconv_7/biasdense/kernel
dense/bias	iterationlearning_rateAdam/m/conv_1/kernelAdam/v/conv_1/kernelAdam/m/conv_1/biasAdam/v/conv_1/biasAdam/m/batchnorm_1/gammaAdam/v/batchnorm_1/gammaAdam/m/batchnorm_1/betaAdam/v/batchnorm_1/betaAdam/m/conv_2/kernelAdam/v/conv_2/kernelAdam/m/conv_2/biasAdam/v/conv_2/biasAdam/m/batchnorm_2/gammaAdam/v/batchnorm_2/gammaAdam/m/batchnorm_2/betaAdam/v/batchnorm_2/betaAdam/m/conv_3/kernelAdam/v/conv_3/kernelAdam/m/conv_3/biasAdam/v/conv_3/biasAdam/m/batchnorm_3/gammaAdam/v/batchnorm_3/gammaAdam/m/batchnorm_3/betaAdam/v/batchnorm_3/betaAdam/m/conv_4/kernelAdam/v/conv_4/kernelAdam/m/conv_4/biasAdam/v/conv_4/biasAdam/m/batchnorm_4/gammaAdam/v/batchnorm_4/gammaAdam/m/batchnorm_4/betaAdam/v/batchnorm_4/betaAdam/m/conv_5/kernelAdam/v/conv_5/kernelAdam/m/conv_5/biasAdam/v/conv_5/biasAdam/m/batchnorm_5/gammaAdam/v/batchnorm_5/gammaAdam/m/batchnorm_5/betaAdam/v/batchnorm_5/betaAdam/m/conv_6/kernelAdam/v/conv_6/kernelAdam/m/conv_6/biasAdam/v/conv_6/biasAdam/m/batchnorm_6/gammaAdam/v/batchnorm_6/gammaAdam/m/batchnorm_6/betaAdam/v/batchnorm_6/betaAdam/m/conv_7/kernelAdam/v/conv_7/kernelAdam/m/conv_7/biasAdam/v/conv_7/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biastotalcount*p
Tini
g2e*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference__traced_restore_710840975��
�	
�
 __inference_loss_fn_10_710840020R
8conv_6_kernel_regularizer_l2loss_readvariableop_resource:
identity��/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp�
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv_6_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_6/kernel/Regularizer/L2LossL2Loss7conv_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_6/kernel/Regularizer/mulMul(conv_6/kernel/Regularizer/mul/x:output:0)conv_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: T
NoOpNoOp0^conv_6/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
E__inference_conv_7_layer_call_and_return_conditional_losses_710839892

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_7/bias/Regularizer/L2Loss/ReadVariableOp�/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
 conv_7/kernel/Regularizer/L2LossL2Loss7conv_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_7/kernel/Regularizer/mulMul(conv_7/kernel/Regularizer/mul/x:output:0)conv_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_7/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_7/bias/Regularizer/L2LossL2Loss5conv_7/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_7/bias/Regularizer/mulMul&conv_7/bias/Regularizer/mul/x:output:0'conv_7/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_7/bias/Regularizer/L2Loss/ReadVariableOp0^conv_7/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_7/bias/Regularizer/L2Loss/ReadVariableOp-conv_7/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
__inference_loss_fn_4_710839972R
8conv_3_kernel_regularizer_l2loss_readvariableop_resource:
identity��/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp�
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv_3_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_3/kernel/Regularizer/L2LossL2Loss7conv_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0)conv_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: T
NoOpNoOp0^conv_3/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
g
K__inference_activation_5_layer_call_and_return_conditional_losses_710838521

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
g
K__inference_activation_2_layer_call_and_return_conditional_losses_710839568

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
�
*__inference_conv_5_layer_call_fn_710839676

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_5_layer_call_and_return_conditional_losses_710838464w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839670:)%
#
_user_specified_name	710839672
�
M
%__inference__update_step_xla_14226563
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv_1_layer_call_and_return_conditional_losses_710838312

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_1/bias/Regularizer/L2Loss/ReadVariableOp�/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_1/kernel/Regularizer/L2LossL2Loss7conv_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0)conv_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_1/bias/Regularizer/L2LossL2Loss5conv_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0'conv_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_1/bias/Regularizer/L2Loss/ReadVariableOp0^conv_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_1/bias/Regularizer/L2Loss/ReadVariableOp-conv_1/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv_1_layer_call_and_return_conditional_losses_710839298

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_1/bias/Regularizer/L2Loss/ReadVariableOp�/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_1/kernel/Regularizer/L2LossL2Loss7conv_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0)conv_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_1/bias/Regularizer/L2LossL2Loss5conv_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0'conv_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_1/bias/Regularizer/L2Loss/ReadVariableOp0^conv_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_1/bias/Regularizer/L2Loss/ReadVariableOp-conv_1/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv_6_layer_call_and_return_conditional_losses_710839793

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_6/bias/Regularizer/L2Loss/ReadVariableOp�/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_6/kernel/Regularizer/L2LossL2Loss7conv_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_6/kernel/Regularizer/mulMul(conv_6/kernel/Regularizer/mul/x:output:0)conv_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_6/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_6/bias/Regularizer/L2LossL2Loss5conv_6/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_6/bias/Regularizer/mulMul&conv_6/bias/Regularizer/mul/x:output:0'conv_6/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_6/bias/Regularizer/L2Loss/ReadVariableOp0^conv_6/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_6/bias/Regularizer/L2Loss/ReadVariableOp-conv_6/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv_5_layer_call_and_return_conditional_losses_710839694

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_5/bias/Regularizer/L2Loss/ReadVariableOp�/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_5/kernel/Regularizer/L2LossL2Loss7conv_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_5/kernel/Regularizer/mulMul(conv_5/kernel/Regularizer/mul/x:output:0)conv_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_5/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_5/bias/Regularizer/L2LossL2Loss5conv_5/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_5/bias/Regularizer/mulMul&conv_5/bias/Regularizer/mul/x:output:0'conv_5/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_5/bias/Regularizer/L2Loss/ReadVariableOp0^conv_5/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_5/bias/Regularizer/L2Loss/ReadVariableOp-conv_5/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710839639

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710839837

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
Y
%__inference__update_step_xla_14226538
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
0__inference_activation_5_layer_call_fn_710839860

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_5_layer_call_and_return_conditional_losses_710838521h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
Y
%__inference__update_step_xla_14226558
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710837938

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
M
%__inference__update_step_xla_14226673
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv_4_layer_call_and_return_conditional_losses_710839595

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_4/bias/Regularizer/L2Loss/ReadVariableOp�/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_4/kernel/Regularizer/L2LossL2Loss7conv_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_4/kernel/Regularizer/mulMul(conv_4/kernel/Regularizer/mul/x:output:0)conv_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_4/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_4/bias/Regularizer/L2LossL2Loss5conv_4/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_4/bias/Regularizer/mulMul&conv_4/bias/Regularizer/mul/x:output:0'conv_4/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_4/bias/Regularizer/L2Loss/ReadVariableOp0^conv_4/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_4/bias/Regularizer/L2Loss/ReadVariableOp-conv_4/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
 __inference_loss_fn_11_710840028D
6conv_6_bias_regularizer_l2loss_readvariableop_resource:
identity��-conv_6/bias/Regularizer/L2Loss/ReadVariableOp�
-conv_6/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6conv_6_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_6/bias/Regularizer/L2LossL2Loss5conv_6/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_6/bias/Regularizer/mulMul&conv_6/bias/Regularizer/mul/x:output:0'conv_6/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_6/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^conv_6/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-conv_6/bias/Regularizer/L2Loss/ReadVariableOp-conv_6/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�

�
/__inference_batchnorm_6_layer_call_fn_710839806

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710838248�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839796:)%
#
_user_specified_name	710839798:)%
#
_user_specified_name	710839800:)%
#
_user_specified_name	710839802
�
M
%__inference__update_step_xla_14226543
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
e
I__inference_activation_layer_call_and_return_conditional_losses_710838331

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
Q
%__inference__update_step_xla_14226668
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:H D

_output_shapes

:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
/__inference_batchnorm_1_layer_call_fn_710839324

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710837956�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839314:)%
#
_user_specified_name	710839316:)%
#
_user_specified_name	710839318:)%
#
_user_specified_name	710839320
�
�
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710838186

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�>
%__inference__traced_restore_710840975
file_prefix8
assignvariableop_conv_1_kernel:,
assignvariableop_1_conv_1_bias:2
$assignvariableop_2_batchnorm_1_gamma:1
#assignvariableop_3_batchnorm_1_beta:8
*assignvariableop_4_batchnorm_1_moving_mean:<
.assignvariableop_5_batchnorm_1_moving_variance::
 assignvariableop_6_conv_2_kernel:,
assignvariableop_7_conv_2_bias:2
$assignvariableop_8_batchnorm_2_gamma:1
#assignvariableop_9_batchnorm_2_beta:9
+assignvariableop_10_batchnorm_2_moving_mean:=
/assignvariableop_11_batchnorm_2_moving_variance:;
!assignvariableop_12_conv_3_kernel:-
assignvariableop_13_conv_3_bias:3
%assignvariableop_14_batchnorm_3_gamma:2
$assignvariableop_15_batchnorm_3_beta:9
+assignvariableop_16_batchnorm_3_moving_mean:=
/assignvariableop_17_batchnorm_3_moving_variance:;
!assignvariableop_18_conv_4_kernel:-
assignvariableop_19_conv_4_bias:3
%assignvariableop_20_batchnorm_4_gamma:2
$assignvariableop_21_batchnorm_4_beta:9
+assignvariableop_22_batchnorm_4_moving_mean:=
/assignvariableop_23_batchnorm_4_moving_variance:;
!assignvariableop_24_conv_5_kernel:-
assignvariableop_25_conv_5_bias:3
%assignvariableop_26_batchnorm_5_gamma:2
$assignvariableop_27_batchnorm_5_beta:9
+assignvariableop_28_batchnorm_5_moving_mean:=
/assignvariableop_29_batchnorm_5_moving_variance:;
!assignvariableop_30_conv_6_kernel:-
assignvariableop_31_conv_6_bias:3
%assignvariableop_32_batchnorm_6_gamma:2
$assignvariableop_33_batchnorm_6_beta:9
+assignvariableop_34_batchnorm_6_moving_mean:=
/assignvariableop_35_batchnorm_6_moving_variance:;
!assignvariableop_36_conv_7_kernel:A-
assignvariableop_37_conv_7_bias:2
 assignvariableop_38_dense_kernel:,
assignvariableop_39_dense_bias:'
assignvariableop_40_iteration:	 +
!assignvariableop_41_learning_rate: B
(assignvariableop_42_adam_m_conv_1_kernel:B
(assignvariableop_43_adam_v_conv_1_kernel:4
&assignvariableop_44_adam_m_conv_1_bias:4
&assignvariableop_45_adam_v_conv_1_bias::
,assignvariableop_46_adam_m_batchnorm_1_gamma::
,assignvariableop_47_adam_v_batchnorm_1_gamma:9
+assignvariableop_48_adam_m_batchnorm_1_beta:9
+assignvariableop_49_adam_v_batchnorm_1_beta:B
(assignvariableop_50_adam_m_conv_2_kernel:B
(assignvariableop_51_adam_v_conv_2_kernel:4
&assignvariableop_52_adam_m_conv_2_bias:4
&assignvariableop_53_adam_v_conv_2_bias::
,assignvariableop_54_adam_m_batchnorm_2_gamma::
,assignvariableop_55_adam_v_batchnorm_2_gamma:9
+assignvariableop_56_adam_m_batchnorm_2_beta:9
+assignvariableop_57_adam_v_batchnorm_2_beta:B
(assignvariableop_58_adam_m_conv_3_kernel:B
(assignvariableop_59_adam_v_conv_3_kernel:4
&assignvariableop_60_adam_m_conv_3_bias:4
&assignvariableop_61_adam_v_conv_3_bias::
,assignvariableop_62_adam_m_batchnorm_3_gamma::
,assignvariableop_63_adam_v_batchnorm_3_gamma:9
+assignvariableop_64_adam_m_batchnorm_3_beta:9
+assignvariableop_65_adam_v_batchnorm_3_beta:B
(assignvariableop_66_adam_m_conv_4_kernel:B
(assignvariableop_67_adam_v_conv_4_kernel:4
&assignvariableop_68_adam_m_conv_4_bias:4
&assignvariableop_69_adam_v_conv_4_bias::
,assignvariableop_70_adam_m_batchnorm_4_gamma::
,assignvariableop_71_adam_v_batchnorm_4_gamma:9
+assignvariableop_72_adam_m_batchnorm_4_beta:9
+assignvariableop_73_adam_v_batchnorm_4_beta:B
(assignvariableop_74_adam_m_conv_5_kernel:B
(assignvariableop_75_adam_v_conv_5_kernel:4
&assignvariableop_76_adam_m_conv_5_bias:4
&assignvariableop_77_adam_v_conv_5_bias::
,assignvariableop_78_adam_m_batchnorm_5_gamma::
,assignvariableop_79_adam_v_batchnorm_5_gamma:9
+assignvariableop_80_adam_m_batchnorm_5_beta:9
+assignvariableop_81_adam_v_batchnorm_5_beta:B
(assignvariableop_82_adam_m_conv_6_kernel:B
(assignvariableop_83_adam_v_conv_6_kernel:4
&assignvariableop_84_adam_m_conv_6_bias:4
&assignvariableop_85_adam_v_conv_6_bias::
,assignvariableop_86_adam_m_batchnorm_6_gamma::
,assignvariableop_87_adam_v_batchnorm_6_gamma:9
+assignvariableop_88_adam_m_batchnorm_6_beta:9
+assignvariableop_89_adam_v_batchnorm_6_beta:B
(assignvariableop_90_adam_m_conv_7_kernel:AB
(assignvariableop_91_adam_v_conv_7_kernel:A4
&assignvariableop_92_adam_m_conv_7_bias:4
&assignvariableop_93_adam_v_conv_7_bias:9
'assignvariableop_94_adam_m_dense_kernel:9
'assignvariableop_95_adam_v_dense_kernel:3
%assignvariableop_96_adam_m_dense_bias:3
%assignvariableop_97_adam_v_dense_bias:#
assignvariableop_98_total: #
assignvariableop_99_count: 
identity_101��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�+
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*�*
value�*B�*eB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*�
value�B�eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*s
dtypesi
g2e	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_batchnorm_1_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_batchnorm_1_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp*assignvariableop_4_batchnorm_1_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batchnorm_1_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_batchnorm_2_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_batchnorm_2_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_batchnorm_2_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batchnorm_2_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv_3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv_3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp%assignvariableop_14_batchnorm_3_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_batchnorm_3_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_batchnorm_3_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batchnorm_3_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv_4_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_conv_4_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_batchnorm_4_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp$assignvariableop_21_batchnorm_4_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_batchnorm_4_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batchnorm_4_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv_5_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_conv_5_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_batchnorm_5_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_batchnorm_5_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp+assignvariableop_28_batchnorm_5_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batchnorm_5_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp!assignvariableop_30_conv_6_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_conv_6_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_batchnorm_6_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp$assignvariableop_33_batchnorm_6_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp+assignvariableop_34_batchnorm_6_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batchnorm_6_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp!assignvariableop_36_conv_7_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_conv_7_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp assignvariableop_38_dense_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_dense_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_iterationIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_learning_rateIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_m_conv_1_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_v_conv_1_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_m_conv_1_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_v_conv_1_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_m_batchnorm_1_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_v_batchnorm_1_gammaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_m_batchnorm_1_betaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_v_batchnorm_1_betaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_conv_2_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_conv_2_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_m_conv_2_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp&assignvariableop_53_adam_v_conv_2_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp,assignvariableop_54_adam_m_batchnorm_2_gammaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_v_batchnorm_2_gammaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp+assignvariableop_56_adam_m_batchnorm_2_betaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_v_batchnorm_2_betaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_m_conv_3_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_v_conv_3_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp&assignvariableop_60_adam_m_conv_3_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp&assignvariableop_61_adam_v_conv_3_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_m_batchnorm_3_gammaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_v_batchnorm_3_gammaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp+assignvariableop_64_adam_m_batchnorm_3_betaIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_v_batchnorm_3_betaIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_m_conv_4_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_v_conv_4_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp&assignvariableop_68_adam_m_conv_4_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp&assignvariableop_69_adam_v_conv_4_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp,assignvariableop_70_adam_m_batchnorm_4_gammaIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_v_batchnorm_4_gammaIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_m_batchnorm_4_betaIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_v_batchnorm_4_betaIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_m_conv_5_kernelIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_v_conv_5_kernelIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp&assignvariableop_76_adam_m_conv_5_biasIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp&assignvariableop_77_adam_v_conv_5_biasIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp,assignvariableop_78_adam_m_batchnorm_5_gammaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp,assignvariableop_79_adam_v_batchnorm_5_gammaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp+assignvariableop_80_adam_m_batchnorm_5_betaIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_v_batchnorm_5_betaIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_m_conv_6_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_v_conv_6_kernelIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp&assignvariableop_84_adam_m_conv_6_biasIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp&assignvariableop_85_adam_v_conv_6_biasIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp,assignvariableop_86_adam_m_batchnorm_6_gammaIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp,assignvariableop_87_adam_v_batchnorm_6_gammaIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp+assignvariableop_88_adam_m_batchnorm_6_betaIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_v_batchnorm_6_betaIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_m_conv_7_kernelIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp(assignvariableop_91_adam_v_conv_7_kernelIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp&assignvariableop_92_adam_m_conv_7_biasIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp&assignvariableop_93_adam_v_conv_7_biasIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp'assignvariableop_94_adam_m_dense_kernelIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp'assignvariableop_95_adam_v_dense_kernelIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp%assignvariableop_96_adam_m_dense_biasIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp%assignvariableop_97_adam_v_dense_biasIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOpassignvariableop_98_totalIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOpassignvariableop_99_countIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_100Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_101IdentityIdentity_100:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_101Identity_101:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_nameconv_1/kernel:+'
%
_user_specified_nameconv_1/bias:1-
+
_user_specified_namebatchnorm_1/gamma:0,
*
_user_specified_namebatchnorm_1/beta:73
1
_user_specified_namebatchnorm_1/moving_mean:;7
5
_user_specified_namebatchnorm_1/moving_variance:-)
'
_user_specified_nameconv_2/kernel:+'
%
_user_specified_nameconv_2/bias:1	-
+
_user_specified_namebatchnorm_2/gamma:0
,
*
_user_specified_namebatchnorm_2/beta:73
1
_user_specified_namebatchnorm_2/moving_mean:;7
5
_user_specified_namebatchnorm_2/moving_variance:-)
'
_user_specified_nameconv_3/kernel:+'
%
_user_specified_nameconv_3/bias:1-
+
_user_specified_namebatchnorm_3/gamma:0,
*
_user_specified_namebatchnorm_3/beta:73
1
_user_specified_namebatchnorm_3/moving_mean:;7
5
_user_specified_namebatchnorm_3/moving_variance:-)
'
_user_specified_nameconv_4/kernel:+'
%
_user_specified_nameconv_4/bias:1-
+
_user_specified_namebatchnorm_4/gamma:0,
*
_user_specified_namebatchnorm_4/beta:73
1
_user_specified_namebatchnorm_4/moving_mean:;7
5
_user_specified_namebatchnorm_4/moving_variance:-)
'
_user_specified_nameconv_5/kernel:+'
%
_user_specified_nameconv_5/bias:1-
+
_user_specified_namebatchnorm_5/gamma:0,
*
_user_specified_namebatchnorm_5/beta:73
1
_user_specified_namebatchnorm_5/moving_mean:;7
5
_user_specified_namebatchnorm_5/moving_variance:-)
'
_user_specified_nameconv_6/kernel:+ '
%
_user_specified_nameconv_6/bias:1!-
+
_user_specified_namebatchnorm_6/gamma:0",
*
_user_specified_namebatchnorm_6/beta:7#3
1
_user_specified_namebatchnorm_6/moving_mean:;$7
5
_user_specified_namebatchnorm_6/moving_variance:-%)
'
_user_specified_nameconv_7/kernel:+&'
%
_user_specified_nameconv_7/bias:,'(
&
_user_specified_namedense/kernel:*(&
$
_user_specified_name
dense/bias:))%
#
_user_specified_name	iteration:-*)
'
_user_specified_namelearning_rate:4+0
.
_user_specified_nameAdam/m/conv_1/kernel:4,0
.
_user_specified_nameAdam/v/conv_1/kernel:2-.
,
_user_specified_nameAdam/m/conv_1/bias:2..
,
_user_specified_nameAdam/v/conv_1/bias:8/4
2
_user_specified_nameAdam/m/batchnorm_1/gamma:804
2
_user_specified_nameAdam/v/batchnorm_1/gamma:713
1
_user_specified_nameAdam/m/batchnorm_1/beta:723
1
_user_specified_nameAdam/v/batchnorm_1/beta:430
.
_user_specified_nameAdam/m/conv_2/kernel:440
.
_user_specified_nameAdam/v/conv_2/kernel:25.
,
_user_specified_nameAdam/m/conv_2/bias:26.
,
_user_specified_nameAdam/v/conv_2/bias:874
2
_user_specified_nameAdam/m/batchnorm_2/gamma:884
2
_user_specified_nameAdam/v/batchnorm_2/gamma:793
1
_user_specified_nameAdam/m/batchnorm_2/beta:7:3
1
_user_specified_nameAdam/v/batchnorm_2/beta:4;0
.
_user_specified_nameAdam/m/conv_3/kernel:4<0
.
_user_specified_nameAdam/v/conv_3/kernel:2=.
,
_user_specified_nameAdam/m/conv_3/bias:2>.
,
_user_specified_nameAdam/v/conv_3/bias:8?4
2
_user_specified_nameAdam/m/batchnorm_3/gamma:8@4
2
_user_specified_nameAdam/v/batchnorm_3/gamma:7A3
1
_user_specified_nameAdam/m/batchnorm_3/beta:7B3
1
_user_specified_nameAdam/v/batchnorm_3/beta:4C0
.
_user_specified_nameAdam/m/conv_4/kernel:4D0
.
_user_specified_nameAdam/v/conv_4/kernel:2E.
,
_user_specified_nameAdam/m/conv_4/bias:2F.
,
_user_specified_nameAdam/v/conv_4/bias:8G4
2
_user_specified_nameAdam/m/batchnorm_4/gamma:8H4
2
_user_specified_nameAdam/v/batchnorm_4/gamma:7I3
1
_user_specified_nameAdam/m/batchnorm_4/beta:7J3
1
_user_specified_nameAdam/v/batchnorm_4/beta:4K0
.
_user_specified_nameAdam/m/conv_5/kernel:4L0
.
_user_specified_nameAdam/v/conv_5/kernel:2M.
,
_user_specified_nameAdam/m/conv_5/bias:2N.
,
_user_specified_nameAdam/v/conv_5/bias:8O4
2
_user_specified_nameAdam/m/batchnorm_5/gamma:8P4
2
_user_specified_nameAdam/v/batchnorm_5/gamma:7Q3
1
_user_specified_nameAdam/m/batchnorm_5/beta:7R3
1
_user_specified_nameAdam/v/batchnorm_5/beta:4S0
.
_user_specified_nameAdam/m/conv_6/kernel:4T0
.
_user_specified_nameAdam/v/conv_6/kernel:2U.
,
_user_specified_nameAdam/m/conv_6/bias:2V.
,
_user_specified_nameAdam/v/conv_6/bias:8W4
2
_user_specified_nameAdam/m/batchnorm_6/gamma:8X4
2
_user_specified_nameAdam/v/batchnorm_6/gamma:7Y3
1
_user_specified_nameAdam/m/batchnorm_6/beta:7Z3
1
_user_specified_nameAdam/v/batchnorm_6/beta:4[0
.
_user_specified_nameAdam/m/conv_7/kernel:4\0
.
_user_specified_nameAdam/v/conv_7/kernel:2].
,
_user_specified_nameAdam/m/conv_7/bias:2^.
,
_user_specified_nameAdam/v/conv_7/bias:3_/
-
_user_specified_nameAdam/m/dense/kernel:3`/
-
_user_specified_nameAdam/v/dense/kernel:1a-
+
_user_specified_nameAdam/m/dense/bias:1b-
+
_user_specified_nameAdam/v/dense/bias:%c!

_user_specified_nametotal:%d!

_user_specified_namecount
�
L
0__inference_activation_1_layer_call_fn_710839464

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_710838369h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
�
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710839738

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
Y
%__inference__update_step_xla_14226658
gradient"
variable:A*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:A: *
	_noinline(:P L
&
_output_shapes
:A
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
M
%__inference__update_step_xla_14226648
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
*__inference_conv_1_layer_call_fn_710839280

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_1_layer_call_and_return_conditional_losses_710838312w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839274:)%
#
_user_specified_name	710839276
�
�
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710838124

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
__inference_loss_fn_2_710839956R
8conv_2_kernel_regularizer_l2loss_readvariableop_resource:
identity��/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp�
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv_2_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_2/kernel/Regularizer/L2LossL2Loss7conv_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0)conv_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: T
NoOpNoOp0^conv_2/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710838204

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv_3_layer_call_and_return_conditional_losses_710839496

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_3/bias/Regularizer/L2Loss/ReadVariableOp�/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_3/kernel/Regularizer/L2LossL2Loss7conv_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0)conv_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_3/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_3/bias/Regularizer/L2LossL2Loss5conv_3/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0'conv_3/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_3/bias/Regularizer/L2Loss/ReadVariableOp0^conv_3/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_3/bias/Regularizer/L2Loss/ReadVariableOp-conv_3/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�"
�	
+__inference_cnnoise_layer_call_fn_710838969
conv_1_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:A

unknown_36:

unknown_37:

unknown_38:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838799w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������A: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������A
&
_user_specified_nameconv_1_input:)%
#
_user_specified_name	710838887:)%
#
_user_specified_name	710838889:)%
#
_user_specified_name	710838891:)%
#
_user_specified_name	710838893:)%
#
_user_specified_name	710838895:)%
#
_user_specified_name	710838897:)%
#
_user_specified_name	710838899:)%
#
_user_specified_name	710838901:)	%
#
_user_specified_name	710838903:)
%
#
_user_specified_name	710838905:)%
#
_user_specified_name	710838907:)%
#
_user_specified_name	710838909:)%
#
_user_specified_name	710838911:)%
#
_user_specified_name	710838913:)%
#
_user_specified_name	710838915:)%
#
_user_specified_name	710838917:)%
#
_user_specified_name	710838919:)%
#
_user_specified_name	710838921:)%
#
_user_specified_name	710838923:)%
#
_user_specified_name	710838925:)%
#
_user_specified_name	710838927:)%
#
_user_specified_name	710838929:)%
#
_user_specified_name	710838931:)%
#
_user_specified_name	710838933:)%
#
_user_specified_name	710838935:)%
#
_user_specified_name	710838937:)%
#
_user_specified_name	710838939:)%
#
_user_specified_name	710838941:)%
#
_user_specified_name	710838943:)%
#
_user_specified_name	710838945:)%
#
_user_specified_name	710838947:) %
#
_user_specified_name	710838949:)!%
#
_user_specified_name	710838951:)"%
#
_user_specified_name	710838953:)#%
#
_user_specified_name	710838955:)$%
#
_user_specified_name	710838957:)%%
#
_user_specified_name	710838959:)&%
#
_user_specified_name	710838961:)'%
#
_user_specified_name	710838963:)(%
#
_user_specified_name	710838965
�
M
%__inference__update_step_xla_14226548
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710839360

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
J
.__inference_activation_layer_call_fn_710839365

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_710838331h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
M
%__inference__update_step_xla_14226583
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
g
K__inference_activation_2_layer_call_and_return_conditional_losses_710838407

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
�
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710837956

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
M
%__inference__update_step_xla_14226588
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710839558

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
M
%__inference__update_step_xla_14226553
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
/__inference_batchnorm_3_layer_call_fn_710839522

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710838080�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839512:)%
#
_user_specified_name	710839514:)%
#
_user_specified_name	710839516:)%
#
_user_specified_name	710839518
�
�
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710838248

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710839459

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
g
K__inference_activation_4_layer_call_and_return_conditional_losses_710838483

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
�
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710838080

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
/__inference_batchnorm_1_layer_call_fn_710839311

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710837938�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839301:)%
#
_user_specified_name	710839303:)%
#
_user_specified_name	710839305:)%
#
_user_specified_name	710839307
�

�
/__inference_batchnorm_2_layer_call_fn_710839410

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710838000�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839400:)%
#
_user_specified_name	710839402:)%
#
_user_specified_name	710839404:)%
#
_user_specified_name	710839406
�
L
0__inference_activation_3_layer_call_fn_710839662

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_3_layer_call_and_return_conditional_losses_710838445h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
M
%__inference__update_step_xla_14226593
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv_2_layer_call_and_return_conditional_losses_710839397

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_2/bias/Regularizer/L2Loss/ReadVariableOp�/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_2/kernel/Regularizer/L2LossL2Loss7conv_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0)conv_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_2/bias/Regularizer/L2LossL2Loss5conv_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0'conv_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_2/bias/Regularizer/L2Loss/ReadVariableOp0^conv_2/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_2/bias/Regularizer/L2Loss/ReadVariableOp-conv_2/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710839540

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
M
%__inference__update_step_xla_14226623
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
*__inference_conv_6_layer_call_fn_710839775

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_6_layer_call_and_return_conditional_losses_710838502w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839769:)%
#
_user_specified_name	710839771
��
�[
"__inference__traced_save_710840666
file_prefix>
$read_disablecopyonread_conv_1_kernel:2
$read_1_disablecopyonread_conv_1_bias:8
*read_2_disablecopyonread_batchnorm_1_gamma:7
)read_3_disablecopyonread_batchnorm_1_beta:>
0read_4_disablecopyonread_batchnorm_1_moving_mean:B
4read_5_disablecopyonread_batchnorm_1_moving_variance:@
&read_6_disablecopyonread_conv_2_kernel:2
$read_7_disablecopyonread_conv_2_bias:8
*read_8_disablecopyonread_batchnorm_2_gamma:7
)read_9_disablecopyonread_batchnorm_2_beta:?
1read_10_disablecopyonread_batchnorm_2_moving_mean:C
5read_11_disablecopyonread_batchnorm_2_moving_variance:A
'read_12_disablecopyonread_conv_3_kernel:3
%read_13_disablecopyonread_conv_3_bias:9
+read_14_disablecopyonread_batchnorm_3_gamma:8
*read_15_disablecopyonread_batchnorm_3_beta:?
1read_16_disablecopyonread_batchnorm_3_moving_mean:C
5read_17_disablecopyonread_batchnorm_3_moving_variance:A
'read_18_disablecopyonread_conv_4_kernel:3
%read_19_disablecopyonread_conv_4_bias:9
+read_20_disablecopyonread_batchnorm_4_gamma:8
*read_21_disablecopyonread_batchnorm_4_beta:?
1read_22_disablecopyonread_batchnorm_4_moving_mean:C
5read_23_disablecopyonread_batchnorm_4_moving_variance:A
'read_24_disablecopyonread_conv_5_kernel:3
%read_25_disablecopyonread_conv_5_bias:9
+read_26_disablecopyonread_batchnorm_5_gamma:8
*read_27_disablecopyonread_batchnorm_5_beta:?
1read_28_disablecopyonread_batchnorm_5_moving_mean:C
5read_29_disablecopyonread_batchnorm_5_moving_variance:A
'read_30_disablecopyonread_conv_6_kernel:3
%read_31_disablecopyonread_conv_6_bias:9
+read_32_disablecopyonread_batchnorm_6_gamma:8
*read_33_disablecopyonread_batchnorm_6_beta:?
1read_34_disablecopyonread_batchnorm_6_moving_mean:C
5read_35_disablecopyonread_batchnorm_6_moving_variance:A
'read_36_disablecopyonread_conv_7_kernel:A3
%read_37_disablecopyonread_conv_7_bias:8
&read_38_disablecopyonread_dense_kernel:2
$read_39_disablecopyonread_dense_bias:-
#read_40_disablecopyonread_iteration:	 1
'read_41_disablecopyonread_learning_rate: H
.read_42_disablecopyonread_adam_m_conv_1_kernel:H
.read_43_disablecopyonread_adam_v_conv_1_kernel::
,read_44_disablecopyonread_adam_m_conv_1_bias::
,read_45_disablecopyonread_adam_v_conv_1_bias:@
2read_46_disablecopyonread_adam_m_batchnorm_1_gamma:@
2read_47_disablecopyonread_adam_v_batchnorm_1_gamma:?
1read_48_disablecopyonread_adam_m_batchnorm_1_beta:?
1read_49_disablecopyonread_adam_v_batchnorm_1_beta:H
.read_50_disablecopyonread_adam_m_conv_2_kernel:H
.read_51_disablecopyonread_adam_v_conv_2_kernel::
,read_52_disablecopyonread_adam_m_conv_2_bias::
,read_53_disablecopyonread_adam_v_conv_2_bias:@
2read_54_disablecopyonread_adam_m_batchnorm_2_gamma:@
2read_55_disablecopyonread_adam_v_batchnorm_2_gamma:?
1read_56_disablecopyonread_adam_m_batchnorm_2_beta:?
1read_57_disablecopyonread_adam_v_batchnorm_2_beta:H
.read_58_disablecopyonread_adam_m_conv_3_kernel:H
.read_59_disablecopyonread_adam_v_conv_3_kernel::
,read_60_disablecopyonread_adam_m_conv_3_bias::
,read_61_disablecopyonread_adam_v_conv_3_bias:@
2read_62_disablecopyonread_adam_m_batchnorm_3_gamma:@
2read_63_disablecopyonread_adam_v_batchnorm_3_gamma:?
1read_64_disablecopyonread_adam_m_batchnorm_3_beta:?
1read_65_disablecopyonread_adam_v_batchnorm_3_beta:H
.read_66_disablecopyonread_adam_m_conv_4_kernel:H
.read_67_disablecopyonread_adam_v_conv_4_kernel::
,read_68_disablecopyonread_adam_m_conv_4_bias::
,read_69_disablecopyonread_adam_v_conv_4_bias:@
2read_70_disablecopyonread_adam_m_batchnorm_4_gamma:@
2read_71_disablecopyonread_adam_v_batchnorm_4_gamma:?
1read_72_disablecopyonread_adam_m_batchnorm_4_beta:?
1read_73_disablecopyonread_adam_v_batchnorm_4_beta:H
.read_74_disablecopyonread_adam_m_conv_5_kernel:H
.read_75_disablecopyonread_adam_v_conv_5_kernel::
,read_76_disablecopyonread_adam_m_conv_5_bias::
,read_77_disablecopyonread_adam_v_conv_5_bias:@
2read_78_disablecopyonread_adam_m_batchnorm_5_gamma:@
2read_79_disablecopyonread_adam_v_batchnorm_5_gamma:?
1read_80_disablecopyonread_adam_m_batchnorm_5_beta:?
1read_81_disablecopyonread_adam_v_batchnorm_5_beta:H
.read_82_disablecopyonread_adam_m_conv_6_kernel:H
.read_83_disablecopyonread_adam_v_conv_6_kernel::
,read_84_disablecopyonread_adam_m_conv_6_bias::
,read_85_disablecopyonread_adam_v_conv_6_bias:@
2read_86_disablecopyonread_adam_m_batchnorm_6_gamma:@
2read_87_disablecopyonread_adam_v_batchnorm_6_gamma:?
1read_88_disablecopyonread_adam_m_batchnorm_6_beta:?
1read_89_disablecopyonread_adam_v_batchnorm_6_beta:H
.read_90_disablecopyonread_adam_m_conv_7_kernel:AH
.read_91_disablecopyonread_adam_v_conv_7_kernel:A:
,read_92_disablecopyonread_adam_m_conv_7_bias::
,read_93_disablecopyonread_adam_v_conv_7_bias:?
-read_94_disablecopyonread_adam_m_dense_kernel:?
-read_95_disablecopyonread_adam_v_dense_kernel:9
+read_96_disablecopyonread_adam_m_dense_bias:9
+read_97_disablecopyonread_adam_v_dense_bias:)
read_98_disablecopyonread_total: )
read_99_disablecopyonread_count: 
savev2_const
identity_201��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOp�Read_96/DisableCopyOnRead�Read_96/ReadVariableOp�Read_97/DisableCopyOnRead�Read_97/ReadVariableOp�Read_98/DisableCopyOnRead�Read_98/ReadVariableOp�Read_99/DisableCopyOnRead�Read_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv_1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv_1_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv_1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv_1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_2/DisableCopyOnReadDisableCopyOnRead*read_2_disablecopyonread_batchnorm_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp*read_2_disablecopyonread_batchnorm_1_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_3/DisableCopyOnReadDisableCopyOnRead)read_3_disablecopyonread_batchnorm_1_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp)read_3_disablecopyonread_batchnorm_1_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead0read_4_disablecopyonread_batchnorm_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp0read_4_disablecopyonread_batchnorm_1_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead4read_5_disablecopyonread_batchnorm_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp4read_5_disablecopyonread_batchnorm_1_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_conv_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_conv_2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_conv_2_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_conv_2_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_8/DisableCopyOnReadDisableCopyOnRead*read_8_disablecopyonread_batchnorm_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp*read_8_disablecopyonread_batchnorm_2_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_batchnorm_2_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_batchnorm_2_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead1read_10_disablecopyonread_batchnorm_2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp1read_10_disablecopyonread_batchnorm_2_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead5read_11_disablecopyonread_batchnorm_2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp5read_11_disablecopyonread_batchnorm_2_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_12/DisableCopyOnReadDisableCopyOnRead'read_12_disablecopyonread_conv_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp'read_12_disablecopyonread_conv_3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_conv_3_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_conv_3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead+read_14_disablecopyonread_batchnorm_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp+read_14_disablecopyonread_batchnorm_3_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_15/DisableCopyOnReadDisableCopyOnRead*read_15_disablecopyonread_batchnorm_3_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp*read_15_disablecopyonread_batchnorm_3_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead1read_16_disablecopyonread_batchnorm_3_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp1read_16_disablecopyonread_batchnorm_3_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnRead5read_17_disablecopyonread_batchnorm_3_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp5read_17_disablecopyonread_batchnorm_3_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_18/DisableCopyOnReadDisableCopyOnRead'read_18_disablecopyonread_conv_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp'read_18_disablecopyonread_conv_4_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_conv_4_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_conv_4_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_20/DisableCopyOnReadDisableCopyOnRead+read_20_disablecopyonread_batchnorm_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp+read_20_disablecopyonread_batchnorm_4_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_21/DisableCopyOnReadDisableCopyOnRead*read_21_disablecopyonread_batchnorm_4_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp*read_21_disablecopyonread_batchnorm_4_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead1read_22_disablecopyonread_batchnorm_4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp1read_22_disablecopyonread_batchnorm_4_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead5read_23_disablecopyonread_batchnorm_4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp5read_23_disablecopyonread_batchnorm_4_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_24/DisableCopyOnReadDisableCopyOnRead'read_24_disablecopyonread_conv_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp'read_24_disablecopyonread_conv_5_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_25/DisableCopyOnReadDisableCopyOnRead%read_25_disablecopyonread_conv_5_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp%read_25_disablecopyonread_conv_5_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_26/DisableCopyOnReadDisableCopyOnRead+read_26_disablecopyonread_batchnorm_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp+read_26_disablecopyonread_batchnorm_5_gamma^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_27/DisableCopyOnReadDisableCopyOnRead*read_27_disablecopyonread_batchnorm_5_beta"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp*read_27_disablecopyonread_batchnorm_5_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_28/DisableCopyOnReadDisableCopyOnRead1read_28_disablecopyonread_batchnorm_5_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp1read_28_disablecopyonread_batchnorm_5_moving_mean^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_29/DisableCopyOnReadDisableCopyOnRead5read_29_disablecopyonread_batchnorm_5_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp5read_29_disablecopyonread_batchnorm_5_moving_variance^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_30/DisableCopyOnReadDisableCopyOnRead'read_30_disablecopyonread_conv_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp'read_30_disablecopyonread_conv_6_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_31/DisableCopyOnReadDisableCopyOnRead%read_31_disablecopyonread_conv_6_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp%read_31_disablecopyonread_conv_6_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_32/DisableCopyOnReadDisableCopyOnRead+read_32_disablecopyonread_batchnorm_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp+read_32_disablecopyonread_batchnorm_6_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_33/DisableCopyOnReadDisableCopyOnRead*read_33_disablecopyonread_batchnorm_6_beta"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp*read_33_disablecopyonread_batchnorm_6_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnRead1read_34_disablecopyonread_batchnorm_6_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp1read_34_disablecopyonread_batchnorm_6_moving_mean^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_35/DisableCopyOnReadDisableCopyOnRead5read_35_disablecopyonread_batchnorm_6_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp5read_35_disablecopyonread_batchnorm_6_moving_variance^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_36/DisableCopyOnReadDisableCopyOnRead'read_36_disablecopyonread_conv_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp'read_36_disablecopyonread_conv_7_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Am
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
:Az
Read_37/DisableCopyOnReadDisableCopyOnRead%read_37_disablecopyonread_conv_7_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp%read_37_disablecopyonread_conv_7_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_38/DisableCopyOnReadDisableCopyOnRead&read_38_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp&read_38_disablecopyonread_dense_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:y
Read_39/DisableCopyOnReadDisableCopyOnRead$read_39_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp$read_39_disablecopyonread_dense_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_40/DisableCopyOnReadDisableCopyOnRead#read_40_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp#read_40_disablecopyonread_iteration^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_41/DisableCopyOnReadDisableCopyOnRead'read_41_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp'read_41_disablecopyonread_learning_rate^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_42/DisableCopyOnReadDisableCopyOnRead.read_42_disablecopyonread_adam_m_conv_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp.read_42_disablecopyonread_adam_m_conv_1_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnRead.read_43_disablecopyonread_adam_v_conv_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp.read_43_disablecopyonread_adam_v_conv_1_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead,read_44_disablecopyonread_adam_m_conv_1_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp,read_44_disablecopyonread_adam_m_conv_1_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnRead,read_45_disablecopyonread_adam_v_conv_1_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp,read_45_disablecopyonread_adam_v_conv_1_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnRead2read_46_disablecopyonread_adam_m_batchnorm_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp2read_46_disablecopyonread_adam_m_batchnorm_1_gamma^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_47/DisableCopyOnReadDisableCopyOnRead2read_47_disablecopyonread_adam_v_batchnorm_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp2read_47_disablecopyonread_adam_v_batchnorm_1_gamma^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_48/DisableCopyOnReadDisableCopyOnRead1read_48_disablecopyonread_adam_m_batchnorm_1_beta"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp1read_48_disablecopyonread_adam_m_batchnorm_1_beta^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_49/DisableCopyOnReadDisableCopyOnRead1read_49_disablecopyonread_adam_v_batchnorm_1_beta"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp1read_49_disablecopyonread_adam_v_batchnorm_1_beta^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_adam_m_conv_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_adam_m_conv_2_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_v_conv_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_v_conv_2_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_52/DisableCopyOnReadDisableCopyOnRead,read_52_disablecopyonread_adam_m_conv_2_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp,read_52_disablecopyonread_adam_m_conv_2_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_53/DisableCopyOnReadDisableCopyOnRead,read_53_disablecopyonread_adam_v_conv_2_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp,read_53_disablecopyonread_adam_v_conv_2_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_54/DisableCopyOnReadDisableCopyOnRead2read_54_disablecopyonread_adam_m_batchnorm_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp2read_54_disablecopyonread_adam_m_batchnorm_2_gamma^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_55/DisableCopyOnReadDisableCopyOnRead2read_55_disablecopyonread_adam_v_batchnorm_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp2read_55_disablecopyonread_adam_v_batchnorm_2_gamma^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_56/DisableCopyOnReadDisableCopyOnRead1read_56_disablecopyonread_adam_m_batchnorm_2_beta"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp1read_56_disablecopyonread_adam_m_batchnorm_2_beta^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_57/DisableCopyOnReadDisableCopyOnRead1read_57_disablecopyonread_adam_v_batchnorm_2_beta"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp1read_57_disablecopyonread_adam_v_batchnorm_2_beta^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_58/DisableCopyOnReadDisableCopyOnRead.read_58_disablecopyonread_adam_m_conv_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp.read_58_disablecopyonread_adam_m_conv_3_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_59/DisableCopyOnReadDisableCopyOnRead.read_59_disablecopyonread_adam_v_conv_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp.read_59_disablecopyonread_adam_v_conv_3_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_60/DisableCopyOnReadDisableCopyOnRead,read_60_disablecopyonread_adam_m_conv_3_bias"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp,read_60_disablecopyonread_adam_m_conv_3_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_61/DisableCopyOnReadDisableCopyOnRead,read_61_disablecopyonread_adam_v_conv_3_bias"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp,read_61_disablecopyonread_adam_v_conv_3_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_62/DisableCopyOnReadDisableCopyOnRead2read_62_disablecopyonread_adam_m_batchnorm_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp2read_62_disablecopyonread_adam_m_batchnorm_3_gamma^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_63/DisableCopyOnReadDisableCopyOnRead2read_63_disablecopyonread_adam_v_batchnorm_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp2read_63_disablecopyonread_adam_v_batchnorm_3_gamma^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_64/DisableCopyOnReadDisableCopyOnRead1read_64_disablecopyonread_adam_m_batchnorm_3_beta"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp1read_64_disablecopyonread_adam_m_batchnorm_3_beta^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_65/DisableCopyOnReadDisableCopyOnRead1read_65_disablecopyonread_adam_v_batchnorm_3_beta"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp1read_65_disablecopyonread_adam_v_batchnorm_3_beta^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_66/DisableCopyOnReadDisableCopyOnRead.read_66_disablecopyonread_adam_m_conv_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp.read_66_disablecopyonread_adam_m_conv_4_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_67/DisableCopyOnReadDisableCopyOnRead.read_67_disablecopyonread_adam_v_conv_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp.read_67_disablecopyonread_adam_v_conv_4_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_68/DisableCopyOnReadDisableCopyOnRead,read_68_disablecopyonread_adam_m_conv_4_bias"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp,read_68_disablecopyonread_adam_m_conv_4_bias^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_69/DisableCopyOnReadDisableCopyOnRead,read_69_disablecopyonread_adam_v_conv_4_bias"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp,read_69_disablecopyonread_adam_v_conv_4_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_70/DisableCopyOnReadDisableCopyOnRead2read_70_disablecopyonread_adam_m_batchnorm_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp2read_70_disablecopyonread_adam_m_batchnorm_4_gamma^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_71/DisableCopyOnReadDisableCopyOnRead2read_71_disablecopyonread_adam_v_batchnorm_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp2read_71_disablecopyonread_adam_v_batchnorm_4_gamma^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_72/DisableCopyOnReadDisableCopyOnRead1read_72_disablecopyonread_adam_m_batchnorm_4_beta"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp1read_72_disablecopyonread_adam_m_batchnorm_4_beta^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_73/DisableCopyOnReadDisableCopyOnRead1read_73_disablecopyonread_adam_v_batchnorm_4_beta"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp1read_73_disablecopyonread_adam_v_batchnorm_4_beta^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_74/DisableCopyOnReadDisableCopyOnRead.read_74_disablecopyonread_adam_m_conv_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp.read_74_disablecopyonread_adam_m_conv_5_kernel^Read_74/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_75/DisableCopyOnReadDisableCopyOnRead.read_75_disablecopyonread_adam_v_conv_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp.read_75_disablecopyonread_adam_v_conv_5_kernel^Read_75/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_76/DisableCopyOnReadDisableCopyOnRead,read_76_disablecopyonread_adam_m_conv_5_bias"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp,read_76_disablecopyonread_adam_m_conv_5_bias^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_77/DisableCopyOnReadDisableCopyOnRead,read_77_disablecopyonread_adam_v_conv_5_bias"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp,read_77_disablecopyonread_adam_v_conv_5_bias^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_78/DisableCopyOnReadDisableCopyOnRead2read_78_disablecopyonread_adam_m_batchnorm_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp2read_78_disablecopyonread_adam_m_batchnorm_5_gamma^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_79/DisableCopyOnReadDisableCopyOnRead2read_79_disablecopyonread_adam_v_batchnorm_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp2read_79_disablecopyonread_adam_v_batchnorm_5_gamma^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_80/DisableCopyOnReadDisableCopyOnRead1read_80_disablecopyonread_adam_m_batchnorm_5_beta"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp1read_80_disablecopyonread_adam_m_batchnorm_5_beta^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_81/DisableCopyOnReadDisableCopyOnRead1read_81_disablecopyonread_adam_v_batchnorm_5_beta"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp1read_81_disablecopyonread_adam_v_batchnorm_5_beta^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_82/DisableCopyOnReadDisableCopyOnRead.read_82_disablecopyonread_adam_m_conv_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp.read_82_disablecopyonread_adam_m_conv_6_kernel^Read_82/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_83/DisableCopyOnReadDisableCopyOnRead.read_83_disablecopyonread_adam_v_conv_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp.read_83_disablecopyonread_adam_v_conv_6_kernel^Read_83/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_84/DisableCopyOnReadDisableCopyOnRead,read_84_disablecopyonread_adam_m_conv_6_bias"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp,read_84_disablecopyonread_adam_m_conv_6_bias^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_85/DisableCopyOnReadDisableCopyOnRead,read_85_disablecopyonread_adam_v_conv_6_bias"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp,read_85_disablecopyonread_adam_v_conv_6_bias^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_86/DisableCopyOnReadDisableCopyOnRead2read_86_disablecopyonread_adam_m_batchnorm_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp2read_86_disablecopyonread_adam_m_batchnorm_6_gamma^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_87/DisableCopyOnReadDisableCopyOnRead2read_87_disablecopyonread_adam_v_batchnorm_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp2read_87_disablecopyonread_adam_v_batchnorm_6_gamma^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_88/DisableCopyOnReadDisableCopyOnRead1read_88_disablecopyonread_adam_m_batchnorm_6_beta"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp1read_88_disablecopyonread_adam_m_batchnorm_6_beta^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_89/DisableCopyOnReadDisableCopyOnRead1read_89_disablecopyonread_adam_v_batchnorm_6_beta"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOp1read_89_disablecopyonread_adam_v_batchnorm_6_beta^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_90/DisableCopyOnReadDisableCopyOnRead.read_90_disablecopyonread_adam_m_conv_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOp.read_90_disablecopyonread_adam_m_conv_7_kernel^Read_90/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_91/DisableCopyOnReadDisableCopyOnRead.read_91_disablecopyonread_adam_v_conv_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOp.read_91_disablecopyonread_adam_v_conv_7_kernel^Read_91/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:A*
dtype0x
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ao
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*&
_output_shapes
:A�
Read_92/DisableCopyOnReadDisableCopyOnRead,read_92_disablecopyonread_adam_m_conv_7_bias"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp,read_92_disablecopyonread_adam_m_conv_7_bias^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_93/DisableCopyOnReadDisableCopyOnRead,read_93_disablecopyonread_adam_v_conv_7_bias"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp,read_93_disablecopyonread_adam_v_conv_7_bias^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_94/DisableCopyOnReadDisableCopyOnRead-read_94_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOp-read_94_disablecopyonread_adam_m_dense_kernel^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_95/DisableCopyOnReadDisableCopyOnRead-read_95_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOp-read_95_disablecopyonread_adam_v_dense_kernel^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_96/DisableCopyOnReadDisableCopyOnRead+read_96_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_96/ReadVariableOpReadVariableOp+read_96_disablecopyonread_adam_m_dense_bias^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_97/DisableCopyOnReadDisableCopyOnRead+read_97_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_97/ReadVariableOpReadVariableOp+read_97_disablecopyonread_adam_v_dense_bias^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_98/DisableCopyOnReadDisableCopyOnReadread_98_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_98/ReadVariableOpReadVariableOpread_98_disablecopyonread_total^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_99/DisableCopyOnReadDisableCopyOnReadread_99_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_99/ReadVariableOpReadVariableOpread_99_disablecopyonread_count^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
: �+
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*�*
value�*B�*eB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*�
value�B�eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *s
dtypesi
g2e	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_200Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_201IdentityIdentity_200:output:0^NoOp*
T0*
_output_shapes
: �)
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_201Identity_201:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_nameconv_1/kernel:+'
%
_user_specified_nameconv_1/bias:1-
+
_user_specified_namebatchnorm_1/gamma:0,
*
_user_specified_namebatchnorm_1/beta:73
1
_user_specified_namebatchnorm_1/moving_mean:;7
5
_user_specified_namebatchnorm_1/moving_variance:-)
'
_user_specified_nameconv_2/kernel:+'
%
_user_specified_nameconv_2/bias:1	-
+
_user_specified_namebatchnorm_2/gamma:0
,
*
_user_specified_namebatchnorm_2/beta:73
1
_user_specified_namebatchnorm_2/moving_mean:;7
5
_user_specified_namebatchnorm_2/moving_variance:-)
'
_user_specified_nameconv_3/kernel:+'
%
_user_specified_nameconv_3/bias:1-
+
_user_specified_namebatchnorm_3/gamma:0,
*
_user_specified_namebatchnorm_3/beta:73
1
_user_specified_namebatchnorm_3/moving_mean:;7
5
_user_specified_namebatchnorm_3/moving_variance:-)
'
_user_specified_nameconv_4/kernel:+'
%
_user_specified_nameconv_4/bias:1-
+
_user_specified_namebatchnorm_4/gamma:0,
*
_user_specified_namebatchnorm_4/beta:73
1
_user_specified_namebatchnorm_4/moving_mean:;7
5
_user_specified_namebatchnorm_4/moving_variance:-)
'
_user_specified_nameconv_5/kernel:+'
%
_user_specified_nameconv_5/bias:1-
+
_user_specified_namebatchnorm_5/gamma:0,
*
_user_specified_namebatchnorm_5/beta:73
1
_user_specified_namebatchnorm_5/moving_mean:;7
5
_user_specified_namebatchnorm_5/moving_variance:-)
'
_user_specified_nameconv_6/kernel:+ '
%
_user_specified_nameconv_6/bias:1!-
+
_user_specified_namebatchnorm_6/gamma:0",
*
_user_specified_namebatchnorm_6/beta:7#3
1
_user_specified_namebatchnorm_6/moving_mean:;$7
5
_user_specified_namebatchnorm_6/moving_variance:-%)
'
_user_specified_nameconv_7/kernel:+&'
%
_user_specified_nameconv_7/bias:,'(
&
_user_specified_namedense/kernel:*(&
$
_user_specified_name
dense/bias:))%
#
_user_specified_name	iteration:-*)
'
_user_specified_namelearning_rate:4+0
.
_user_specified_nameAdam/m/conv_1/kernel:4,0
.
_user_specified_nameAdam/v/conv_1/kernel:2-.
,
_user_specified_nameAdam/m/conv_1/bias:2..
,
_user_specified_nameAdam/v/conv_1/bias:8/4
2
_user_specified_nameAdam/m/batchnorm_1/gamma:804
2
_user_specified_nameAdam/v/batchnorm_1/gamma:713
1
_user_specified_nameAdam/m/batchnorm_1/beta:723
1
_user_specified_nameAdam/v/batchnorm_1/beta:430
.
_user_specified_nameAdam/m/conv_2/kernel:440
.
_user_specified_nameAdam/v/conv_2/kernel:25.
,
_user_specified_nameAdam/m/conv_2/bias:26.
,
_user_specified_nameAdam/v/conv_2/bias:874
2
_user_specified_nameAdam/m/batchnorm_2/gamma:884
2
_user_specified_nameAdam/v/batchnorm_2/gamma:793
1
_user_specified_nameAdam/m/batchnorm_2/beta:7:3
1
_user_specified_nameAdam/v/batchnorm_2/beta:4;0
.
_user_specified_nameAdam/m/conv_3/kernel:4<0
.
_user_specified_nameAdam/v/conv_3/kernel:2=.
,
_user_specified_nameAdam/m/conv_3/bias:2>.
,
_user_specified_nameAdam/v/conv_3/bias:8?4
2
_user_specified_nameAdam/m/batchnorm_3/gamma:8@4
2
_user_specified_nameAdam/v/batchnorm_3/gamma:7A3
1
_user_specified_nameAdam/m/batchnorm_3/beta:7B3
1
_user_specified_nameAdam/v/batchnorm_3/beta:4C0
.
_user_specified_nameAdam/m/conv_4/kernel:4D0
.
_user_specified_nameAdam/v/conv_4/kernel:2E.
,
_user_specified_nameAdam/m/conv_4/bias:2F.
,
_user_specified_nameAdam/v/conv_4/bias:8G4
2
_user_specified_nameAdam/m/batchnorm_4/gamma:8H4
2
_user_specified_nameAdam/v/batchnorm_4/gamma:7I3
1
_user_specified_nameAdam/m/batchnorm_4/beta:7J3
1
_user_specified_nameAdam/v/batchnorm_4/beta:4K0
.
_user_specified_nameAdam/m/conv_5/kernel:4L0
.
_user_specified_nameAdam/v/conv_5/kernel:2M.
,
_user_specified_nameAdam/m/conv_5/bias:2N.
,
_user_specified_nameAdam/v/conv_5/bias:8O4
2
_user_specified_nameAdam/m/batchnorm_5/gamma:8P4
2
_user_specified_nameAdam/v/batchnorm_5/gamma:7Q3
1
_user_specified_nameAdam/m/batchnorm_5/beta:7R3
1
_user_specified_nameAdam/v/batchnorm_5/beta:4S0
.
_user_specified_nameAdam/m/conv_6/kernel:4T0
.
_user_specified_nameAdam/v/conv_6/kernel:2U.
,
_user_specified_nameAdam/m/conv_6/bias:2V.
,
_user_specified_nameAdam/v/conv_6/bias:8W4
2
_user_specified_nameAdam/m/batchnorm_6/gamma:8X4
2
_user_specified_nameAdam/v/batchnorm_6/gamma:7Y3
1
_user_specified_nameAdam/m/batchnorm_6/beta:7Z3
1
_user_specified_nameAdam/v/batchnorm_6/beta:4[0
.
_user_specified_nameAdam/m/conv_7/kernel:4\0
.
_user_specified_nameAdam/v/conv_7/kernel:2].
,
_user_specified_nameAdam/m/conv_7/bias:2^.
,
_user_specified_nameAdam/v/conv_7/bias:3_/
-
_user_specified_nameAdam/m/dense/kernel:3`/
-
_user_specified_nameAdam/v/dense/kernel:1a-
+
_user_specified_nameAdam/m/dense/bias:1b-
+
_user_specified_nameAdam/v/dense/bias:%c!

_user_specified_nametotal:%d!

_user_specified_namecount:=e9

_output_shapes
: 

_user_specified_nameConst
�
M
%__inference__update_step_xla_14226628
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
/__inference_batchnorm_4_layer_call_fn_710839621

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710838142�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839611:)%
#
_user_specified_name	710839613:)%
#
_user_specified_name	710839615:)%
#
_user_specified_name	710839617
�
M
%__inference__update_step_xla_14226643
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
0__inference_activation_4_layer_call_fn_710839761

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_4_layer_call_and_return_conditional_losses_710838483h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
g
K__inference_activation_3_layer_call_and_return_conditional_losses_710839667

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�!
�	
'__inference_signature_wrapper_710839215
conv_1_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:A

unknown_36:

unknown_37:

unknown_38:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference__wrapped_model_710837920w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������A: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������A
&
_user_specified_nameconv_1_input:)%
#
_user_specified_name	710839133:)%
#
_user_specified_name	710839135:)%
#
_user_specified_name	710839137:)%
#
_user_specified_name	710839139:)%
#
_user_specified_name	710839141:)%
#
_user_specified_name	710839143:)%
#
_user_specified_name	710839145:)%
#
_user_specified_name	710839147:)	%
#
_user_specified_name	710839149:)
%
#
_user_specified_name	710839151:)%
#
_user_specified_name	710839153:)%
#
_user_specified_name	710839155:)%
#
_user_specified_name	710839157:)%
#
_user_specified_name	710839159:)%
#
_user_specified_name	710839161:)%
#
_user_specified_name	710839163:)%
#
_user_specified_name	710839165:)%
#
_user_specified_name	710839167:)%
#
_user_specified_name	710839169:)%
#
_user_specified_name	710839171:)%
#
_user_specified_name	710839173:)%
#
_user_specified_name	710839175:)%
#
_user_specified_name	710839177:)%
#
_user_specified_name	710839179:)%
#
_user_specified_name	710839181:)%
#
_user_specified_name	710839183:)%
#
_user_specified_name	710839185:)%
#
_user_specified_name	710839187:)%
#
_user_specified_name	710839189:)%
#
_user_specified_name	710839191:)%
#
_user_specified_name	710839193:) %
#
_user_specified_name	710839195:)!%
#
_user_specified_name	710839197:)"%
#
_user_specified_name	710839199:)#%
#
_user_specified_name	710839201:)$%
#
_user_specified_name	710839203:)%%
#
_user_specified_name	710839205:)&%
#
_user_specified_name	710839207:)'%
#
_user_specified_name	710839209:)(%
#
_user_specified_name	710839211
�
�
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710839855

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
Y
%__inference__update_step_xla_14226638
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
Y
%__inference__update_step_xla_14226578
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv_2_layer_call_and_return_conditional_losses_710838350

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_2/bias/Regularizer/L2Loss/ReadVariableOp�/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_2/kernel/Regularizer/L2LossL2Loss7conv_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0)conv_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_2/bias/Regularizer/L2LossL2Loss5conv_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0'conv_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_2/bias/Regularizer/L2Loss/ReadVariableOp0^conv_2/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_2/bias/Regularizer/L2Loss/ReadVariableOp-conv_2/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710839342

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710839441

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv_7_layer_call_and_return_conditional_losses_710838540

inputs8
conv2d_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_7/bias/Regularizer/L2Loss/ReadVariableOp�/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
 conv_7/kernel/Regularizer/L2LossL2Loss7conv_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_7/kernel/Regularizer/mulMul(conv_7/kernel/Regularizer/mul/x:output:0)conv_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_7/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_7/bias/Regularizer/L2LossL2Loss5conv_7/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_7/bias/Regularizer/mulMul&conv_7/bias/Regularizer/mul/x:output:0'conv_7/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_7/bias/Regularizer/L2Loss/ReadVariableOp0^conv_7/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_7/bias/Regularizer/L2Loss/ReadVariableOp-conv_7/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
g
K__inference_activation_5_layer_call_and_return_conditional_losses_710839865

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�

�
/__inference_batchnorm_2_layer_call_fn_710839423

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710838018�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839413:)%
#
_user_specified_name	710839415:)%
#
_user_specified_name	710839417:)%
#
_user_specified_name	710839419
�
�
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710838142

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
__inference_loss_fn_8_710840004R
8conv_5_kernel_regularizer_l2loss_readvariableop_resource:
identity��/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp�
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv_5_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_5/kernel/Regularizer/L2LossL2Loss7conv_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_5/kernel/Regularizer/mulMul(conv_5/kernel/Regularizer/mul/x:output:0)conv_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_5/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: T
NoOpNoOp0^conv_5/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�

�
/__inference_batchnorm_5_layer_call_fn_710839720

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710838204�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839710:)%
#
_user_specified_name	710839712:)%
#
_user_specified_name	710839714:)%
#
_user_specified_name	710839716
�
M
%__inference__update_step_xla_14226573
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
D__inference_dense_layer_call_and_return_conditional_losses_710839932

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:���������A�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:���������Ar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������Ab
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������AV
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv_4_layer_call_and_return_conditional_losses_710838426

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_4/bias/Regularizer/L2Loss/ReadVariableOp�/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_4/kernel/Regularizer/L2LossL2Loss7conv_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_4/kernel/Regularizer/mulMul(conv_4/kernel/Regularizer/mul/x:output:0)conv_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_4/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_4/bias/Regularizer/L2LossL2Loss5conv_4/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_4/bias/Regularizer/mulMul&conv_4/bias/Regularizer/mul/x:output:0'conv_4/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_4/bias/Regularizer/L2Loss/ReadVariableOp0^conv_4/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_4/bias/Regularizer/L2Loss/ReadVariableOp-conv_4/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
__inference_loss_fn_5_710839980D
6conv_3_bias_regularizer_l2loss_readvariableop_resource:
identity��-conv_3/bias/Regularizer/L2Loss/ReadVariableOp�
-conv_3/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6conv_3_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_3/bias/Regularizer/L2LossL2Loss5conv_3/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0'conv_3/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_3/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^conv_3/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-conv_3/bias/Regularizer/L2Loss/ReadVariableOp-conv_3/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710838018

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_conv_2_layer_call_fn_710839379

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_2_layer_call_and_return_conditional_losses_710838350w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839373:)%
#
_user_specified_name	710839375
��
�
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838799
conv_1_input*
conv_1_710838642:
conv_1_710838644:#
batchnorm_1_710838647:#
batchnorm_1_710838649:#
batchnorm_1_710838651:#
batchnorm_1_710838653:*
conv_2_710838657:
conv_2_710838659:#
batchnorm_2_710838662:#
batchnorm_2_710838664:#
batchnorm_2_710838666:#
batchnorm_2_710838668:*
conv_3_710838672:
conv_3_710838674:#
batchnorm_3_710838677:#
batchnorm_3_710838679:#
batchnorm_3_710838681:#
batchnorm_3_710838683:*
conv_4_710838687:
conv_4_710838689:#
batchnorm_4_710838692:#
batchnorm_4_710838694:#
batchnorm_4_710838696:#
batchnorm_4_710838698:*
conv_5_710838702:
conv_5_710838704:#
batchnorm_5_710838707:#
batchnorm_5_710838709:#
batchnorm_5_710838711:#
batchnorm_5_710838713:*
conv_6_710838717:
conv_6_710838719:#
batchnorm_6_710838722:#
batchnorm_6_710838724:#
batchnorm_6_710838726:#
batchnorm_6_710838728:*
conv_7_710838732:A
conv_7_710838734:!
dense_710838737:
dense_710838739:
identity��#batchnorm_1/StatefulPartitionedCall�#batchnorm_2/StatefulPartitionedCall�#batchnorm_3/StatefulPartitionedCall�#batchnorm_4/StatefulPartitionedCall�#batchnorm_5/StatefulPartitionedCall�#batchnorm_6/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�-conv_1/bias/Regularizer/L2Loss/ReadVariableOp�/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp�conv_2/StatefulPartitionedCall�-conv_2/bias/Regularizer/L2Loss/ReadVariableOp�/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp�conv_3/StatefulPartitionedCall�-conv_3/bias/Regularizer/L2Loss/ReadVariableOp�/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp�conv_4/StatefulPartitionedCall�-conv_4/bias/Regularizer/L2Loss/ReadVariableOp�/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp�conv_5/StatefulPartitionedCall�-conv_5/bias/Regularizer/L2Loss/ReadVariableOp�/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp�conv_6/StatefulPartitionedCall�-conv_6/bias/Regularizer/L2Loss/ReadVariableOp�/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp�conv_7/StatefulPartitionedCall�-conv_7/bias/Regularizer/L2Loss/ReadVariableOp�/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCallconv_1_inputconv_1_710838642conv_1_710838644*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_1_layer_call_and_return_conditional_losses_710838312�
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_710838647batchnorm_1_710838649batchnorm_1_710838651batchnorm_1_710838653*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710837956�
activation/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_710838331�
conv_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv_2_710838657conv_2_710838659*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_2_layer_call_and_return_conditional_losses_710838350�
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_710838662batchnorm_2_710838664batchnorm_2_710838666batchnorm_2_710838668*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710838018�
activation_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_710838369�
conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv_3_710838672conv_3_710838674*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_3_layer_call_and_return_conditional_losses_710838388�
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_710838677batchnorm_3_710838679batchnorm_3_710838681batchnorm_3_710838683*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710838080�
activation_2/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_2_layer_call_and_return_conditional_losses_710838407�
conv_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv_4_710838687conv_4_710838689*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_4_layer_call_and_return_conditional_losses_710838426�
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batchnorm_4_710838692batchnorm_4_710838694batchnorm_4_710838696batchnorm_4_710838698*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710838142�
activation_3/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_3_layer_call_and_return_conditional_losses_710838445�
conv_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv_5_710838702conv_5_710838704*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_5_layer_call_and_return_conditional_losses_710838464�
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batchnorm_5_710838707batchnorm_5_710838709batchnorm_5_710838711batchnorm_5_710838713*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710838204�
activation_4/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_4_layer_call_and_return_conditional_losses_710838483�
conv_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv_6_710838717conv_6_710838719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_6_layer_call_and_return_conditional_losses_710838502�
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batchnorm_6_710838722batchnorm_6_710838724batchnorm_6_710838726batchnorm_6_710838728*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710838266�
activation_5/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_5_layer_call_and_return_conditional_losses_710838521�
conv_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv_7_710838732conv_7_710838734*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_7_layer_call_and_return_conditional_losses_710838540�
dense/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0dense_710838737dense_710838739*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_710838576�
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_1_710838642*&
_output_shapes
:*
dtype0�
 conv_1/kernel/Regularizer/L2LossL2Loss7conv_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0)conv_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_1_710838644*
_output_shapes
:*
dtype0�
conv_1/bias/Regularizer/L2LossL2Loss5conv_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0'conv_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_2_710838657*&
_output_shapes
:*
dtype0�
 conv_2/kernel/Regularizer/L2LossL2Loss7conv_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0)conv_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_2_710838659*
_output_shapes
:*
dtype0�
conv_2/bias/Regularizer/L2LossL2Loss5conv_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0'conv_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_3_710838672*&
_output_shapes
:*
dtype0�
 conv_3/kernel/Regularizer/L2LossL2Loss7conv_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0)conv_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_3/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_3_710838674*
_output_shapes
:*
dtype0�
conv_3/bias/Regularizer/L2LossL2Loss5conv_3/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0'conv_3/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_4_710838687*&
_output_shapes
:*
dtype0�
 conv_4/kernel/Regularizer/L2LossL2Loss7conv_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_4/kernel/Regularizer/mulMul(conv_4/kernel/Regularizer/mul/x:output:0)conv_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_4/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_4_710838689*
_output_shapes
:*
dtype0�
conv_4/bias/Regularizer/L2LossL2Loss5conv_4/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_4/bias/Regularizer/mulMul&conv_4/bias/Regularizer/mul/x:output:0'conv_4/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_5_710838702*&
_output_shapes
:*
dtype0�
 conv_5/kernel/Regularizer/L2LossL2Loss7conv_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_5/kernel/Regularizer/mulMul(conv_5/kernel/Regularizer/mul/x:output:0)conv_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_5/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_5_710838704*
_output_shapes
:*
dtype0�
conv_5/bias/Regularizer/L2LossL2Loss5conv_5/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_5/bias/Regularizer/mulMul&conv_5/bias/Regularizer/mul/x:output:0'conv_5/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_6_710838717*&
_output_shapes
:*
dtype0�
 conv_6/kernel/Regularizer/L2LossL2Loss7conv_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_6/kernel/Regularizer/mulMul(conv_6/kernel/Regularizer/mul/x:output:0)conv_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_6/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_6_710838719*
_output_shapes
:*
dtype0�
conv_6/bias/Regularizer/L2LossL2Loss5conv_6/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_6/bias/Regularizer/mulMul&conv_6/bias/Regularizer/mul/x:output:0'conv_6/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_7_710838732*&
_output_shapes
:A*
dtype0�
 conv_7/kernel/Regularizer/L2LossL2Loss7conv_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_7/kernel/Regularizer/mulMul(conv_7/kernel/Regularizer/mul/x:output:0)conv_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_7/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_7_710838734*
_output_shapes
:*
dtype0�
conv_7/bias/Regularizer/L2LossL2Loss5conv_7/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_7/bias/Regularizer/mulMul&conv_7/bias/Regularizer/mul/x:output:0'conv_7/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: }
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A�	
NoOpNoOp$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall^conv_1/StatefulPartitionedCall.^conv_1/bias/Regularizer/L2Loss/ReadVariableOp0^conv_1/kernel/Regularizer/L2Loss/ReadVariableOp^conv_2/StatefulPartitionedCall.^conv_2/bias/Regularizer/L2Loss/ReadVariableOp0^conv_2/kernel/Regularizer/L2Loss/ReadVariableOp^conv_3/StatefulPartitionedCall.^conv_3/bias/Regularizer/L2Loss/ReadVariableOp0^conv_3/kernel/Regularizer/L2Loss/ReadVariableOp^conv_4/StatefulPartitionedCall.^conv_4/bias/Regularizer/L2Loss/ReadVariableOp0^conv_4/kernel/Regularizer/L2Loss/ReadVariableOp^conv_5/StatefulPartitionedCall.^conv_5/bias/Regularizer/L2Loss/ReadVariableOp0^conv_5/kernel/Regularizer/L2Loss/ReadVariableOp^conv_6/StatefulPartitionedCall.^conv_6/bias/Regularizer/L2Loss/ReadVariableOp0^conv_6/kernel/Regularizer/L2Loss/ReadVariableOp^conv_7/StatefulPartitionedCall.^conv_7/bias/Regularizer/L2Loss/ReadVariableOp0^conv_7/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������A: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2^
-conv_1/bias/Regularizer/L2Loss/ReadVariableOp-conv_1/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2^
-conv_2/bias/Regularizer/L2Loss/ReadVariableOp-conv_2/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2^
-conv_3/bias/Regularizer/L2Loss/ReadVariableOp-conv_3/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2^
-conv_4/bias/Regularizer/L2Loss/ReadVariableOp-conv_4/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2^
-conv_5/bias/Regularizer/L2Loss/ReadVariableOp-conv_5/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2^
-conv_6/bias/Regularizer/L2Loss/ReadVariableOp-conv_6/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2^
-conv_7/bias/Regularizer/L2Loss/ReadVariableOp-conv_7/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
/
_output_shapes
:���������A
&
_user_specified_nameconv_1_input:)%
#
_user_specified_name	710838642:)%
#
_user_specified_name	710838644:)%
#
_user_specified_name	710838647:)%
#
_user_specified_name	710838649:)%
#
_user_specified_name	710838651:)%
#
_user_specified_name	710838653:)%
#
_user_specified_name	710838657:)%
#
_user_specified_name	710838659:)	%
#
_user_specified_name	710838662:)
%
#
_user_specified_name	710838664:)%
#
_user_specified_name	710838666:)%
#
_user_specified_name	710838668:)%
#
_user_specified_name	710838672:)%
#
_user_specified_name	710838674:)%
#
_user_specified_name	710838677:)%
#
_user_specified_name	710838679:)%
#
_user_specified_name	710838681:)%
#
_user_specified_name	710838683:)%
#
_user_specified_name	710838687:)%
#
_user_specified_name	710838689:)%
#
_user_specified_name	710838692:)%
#
_user_specified_name	710838694:)%
#
_user_specified_name	710838696:)%
#
_user_specified_name	710838698:)%
#
_user_specified_name	710838702:)%
#
_user_specified_name	710838704:)%
#
_user_specified_name	710838707:)%
#
_user_specified_name	710838709:)%
#
_user_specified_name	710838711:)%
#
_user_specified_name	710838713:)%
#
_user_specified_name	710838717:) %
#
_user_specified_name	710838719:)!%
#
_user_specified_name	710838722:)"%
#
_user_specified_name	710838724:)#%
#
_user_specified_name	710838726:)$%
#
_user_specified_name	710838728:)%%
#
_user_specified_name	710838732:)&%
#
_user_specified_name	710838734:)'%
#
_user_specified_name	710838737:)(%
#
_user_specified_name	710838739
�
�
*__inference_conv_3_layer_call_fn_710839478

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_3_layer_call_and_return_conditional_losses_710838388w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839472:)%
#
_user_specified_name	710839474
�
M
%__inference__update_step_xla_14226653
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
��
�#
$__inference__wrapped_model_710837920
conv_1_inputG
-cnnoise_conv_1_conv2d_readvariableop_resource:<
.cnnoise_conv_1_biasadd_readvariableop_resource:9
+cnnoise_batchnorm_1_readvariableop_resource:;
-cnnoise_batchnorm_1_readvariableop_1_resource:J
<cnnoise_batchnorm_1_fusedbatchnormv3_readvariableop_resource:L
>cnnoise_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource:G
-cnnoise_conv_2_conv2d_readvariableop_resource:<
.cnnoise_conv_2_biasadd_readvariableop_resource:9
+cnnoise_batchnorm_2_readvariableop_resource:;
-cnnoise_batchnorm_2_readvariableop_1_resource:J
<cnnoise_batchnorm_2_fusedbatchnormv3_readvariableop_resource:L
>cnnoise_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource:G
-cnnoise_conv_3_conv2d_readvariableop_resource:<
.cnnoise_conv_3_biasadd_readvariableop_resource:9
+cnnoise_batchnorm_3_readvariableop_resource:;
-cnnoise_batchnorm_3_readvariableop_1_resource:J
<cnnoise_batchnorm_3_fusedbatchnormv3_readvariableop_resource:L
>cnnoise_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource:G
-cnnoise_conv_4_conv2d_readvariableop_resource:<
.cnnoise_conv_4_biasadd_readvariableop_resource:9
+cnnoise_batchnorm_4_readvariableop_resource:;
-cnnoise_batchnorm_4_readvariableop_1_resource:J
<cnnoise_batchnorm_4_fusedbatchnormv3_readvariableop_resource:L
>cnnoise_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource:G
-cnnoise_conv_5_conv2d_readvariableop_resource:<
.cnnoise_conv_5_biasadd_readvariableop_resource:9
+cnnoise_batchnorm_5_readvariableop_resource:;
-cnnoise_batchnorm_5_readvariableop_1_resource:J
<cnnoise_batchnorm_5_fusedbatchnormv3_readvariableop_resource:L
>cnnoise_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource:G
-cnnoise_conv_6_conv2d_readvariableop_resource:<
.cnnoise_conv_6_biasadd_readvariableop_resource:9
+cnnoise_batchnorm_6_readvariableop_resource:;
-cnnoise_batchnorm_6_readvariableop_1_resource:J
<cnnoise_batchnorm_6_fusedbatchnormv3_readvariableop_resource:L
>cnnoise_batchnorm_6_fusedbatchnormv3_readvariableop_1_resource:G
-cnnoise_conv_7_conv2d_readvariableop_resource:A<
.cnnoise_conv_7_biasadd_readvariableop_resource:A
/cnnoise_dense_tensordot_readvariableop_resource:;
-cnnoise_dense_biasadd_readvariableop_resource:
identity��3cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp�5cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1�"cnnoise/batchnorm_1/ReadVariableOp�$cnnoise/batchnorm_1/ReadVariableOp_1�3cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp�5cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1�"cnnoise/batchnorm_2/ReadVariableOp�$cnnoise/batchnorm_2/ReadVariableOp_1�3cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp�5cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1�"cnnoise/batchnorm_3/ReadVariableOp�$cnnoise/batchnorm_3/ReadVariableOp_1�3cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp�5cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1�"cnnoise/batchnorm_4/ReadVariableOp�$cnnoise/batchnorm_4/ReadVariableOp_1�3cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp�5cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1�"cnnoise/batchnorm_5/ReadVariableOp�$cnnoise/batchnorm_5/ReadVariableOp_1�3cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp�5cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1�"cnnoise/batchnorm_6/ReadVariableOp�$cnnoise/batchnorm_6/ReadVariableOp_1�%cnnoise/conv_1/BiasAdd/ReadVariableOp�$cnnoise/conv_1/Conv2D/ReadVariableOp�%cnnoise/conv_2/BiasAdd/ReadVariableOp�$cnnoise/conv_2/Conv2D/ReadVariableOp�%cnnoise/conv_3/BiasAdd/ReadVariableOp�$cnnoise/conv_3/Conv2D/ReadVariableOp�%cnnoise/conv_4/BiasAdd/ReadVariableOp�$cnnoise/conv_4/Conv2D/ReadVariableOp�%cnnoise/conv_5/BiasAdd/ReadVariableOp�$cnnoise/conv_5/Conv2D/ReadVariableOp�%cnnoise/conv_6/BiasAdd/ReadVariableOp�$cnnoise/conv_6/Conv2D/ReadVariableOp�%cnnoise/conv_7/BiasAdd/ReadVariableOp�$cnnoise/conv_7/Conv2D/ReadVariableOp�$cnnoise/dense/BiasAdd/ReadVariableOp�&cnnoise/dense/Tensordot/ReadVariableOp�
$cnnoise/conv_1/Conv2D/ReadVariableOpReadVariableOp-cnnoise_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
cnnoise/conv_1/Conv2DConv2Dconv_1_input,cnnoise/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
�
%cnnoise/conv_1/BiasAdd/ReadVariableOpReadVariableOp.cnnoise_conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cnnoise/conv_1/BiasAddBiasAddcnnoise/conv_1/Conv2D:output:0-cnnoise/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
"cnnoise/batchnorm_1/ReadVariableOpReadVariableOp+cnnoise_batchnorm_1_readvariableop_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_1/ReadVariableOp_1ReadVariableOp-cnnoise_batchnorm_1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp<cnnoise_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>cnnoise_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_1/FusedBatchNormV3FusedBatchNormV3cnnoise/conv_1/BiasAdd:output:0*cnnoise/batchnorm_1/ReadVariableOp:value:0,cnnoise/batchnorm_1/ReadVariableOp_1:value:0;cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0=cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������A:::::*
epsilon%o�:*
is_training( �
cnnoise/activation/ReluRelu(cnnoise/batchnorm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������A�
$cnnoise/conv_2/Conv2D/ReadVariableOpReadVariableOp-cnnoise_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
cnnoise/conv_2/Conv2DConv2D%cnnoise/activation/Relu:activations:0,cnnoise/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
�
%cnnoise/conv_2/BiasAdd/ReadVariableOpReadVariableOp.cnnoise_conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cnnoise/conv_2/BiasAddBiasAddcnnoise/conv_2/Conv2D:output:0-cnnoise/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
"cnnoise/batchnorm_2/ReadVariableOpReadVariableOp+cnnoise_batchnorm_2_readvariableop_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_2/ReadVariableOp_1ReadVariableOp-cnnoise_batchnorm_2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp<cnnoise_batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>cnnoise_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_2/FusedBatchNormV3FusedBatchNormV3cnnoise/conv_2/BiasAdd:output:0*cnnoise/batchnorm_2/ReadVariableOp:value:0,cnnoise/batchnorm_2/ReadVariableOp_1:value:0;cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:0=cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������A:::::*
epsilon%o�:*
is_training( �
cnnoise/activation_1/ReluRelu(cnnoise/batchnorm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������A�
$cnnoise/conv_3/Conv2D/ReadVariableOpReadVariableOp-cnnoise_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
cnnoise/conv_3/Conv2DConv2D'cnnoise/activation_1/Relu:activations:0,cnnoise/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
�
%cnnoise/conv_3/BiasAdd/ReadVariableOpReadVariableOp.cnnoise_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cnnoise/conv_3/BiasAddBiasAddcnnoise/conv_3/Conv2D:output:0-cnnoise/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
"cnnoise/batchnorm_3/ReadVariableOpReadVariableOp+cnnoise_batchnorm_3_readvariableop_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_3/ReadVariableOp_1ReadVariableOp-cnnoise_batchnorm_3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp<cnnoise_batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>cnnoise_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_3/FusedBatchNormV3FusedBatchNormV3cnnoise/conv_3/BiasAdd:output:0*cnnoise/batchnorm_3/ReadVariableOp:value:0,cnnoise/batchnorm_3/ReadVariableOp_1:value:0;cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:0=cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������A:::::*
epsilon%o�:*
is_training( �
cnnoise/activation_2/ReluRelu(cnnoise/batchnorm_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������A�
$cnnoise/conv_4/Conv2D/ReadVariableOpReadVariableOp-cnnoise_conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
cnnoise/conv_4/Conv2DConv2D'cnnoise/activation_2/Relu:activations:0,cnnoise/conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
�
%cnnoise/conv_4/BiasAdd/ReadVariableOpReadVariableOp.cnnoise_conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cnnoise/conv_4/BiasAddBiasAddcnnoise/conv_4/Conv2D:output:0-cnnoise/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
"cnnoise/batchnorm_4/ReadVariableOpReadVariableOp+cnnoise_batchnorm_4_readvariableop_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_4/ReadVariableOp_1ReadVariableOp-cnnoise_batchnorm_4_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp<cnnoise_batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>cnnoise_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_4/FusedBatchNormV3FusedBatchNormV3cnnoise/conv_4/BiasAdd:output:0*cnnoise/batchnorm_4/ReadVariableOp:value:0,cnnoise/batchnorm_4/ReadVariableOp_1:value:0;cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:0=cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������A:::::*
epsilon%o�:*
is_training( �
cnnoise/activation_3/ReluRelu(cnnoise/batchnorm_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������A�
$cnnoise/conv_5/Conv2D/ReadVariableOpReadVariableOp-cnnoise_conv_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
cnnoise/conv_5/Conv2DConv2D'cnnoise/activation_3/Relu:activations:0,cnnoise/conv_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
�
%cnnoise/conv_5/BiasAdd/ReadVariableOpReadVariableOp.cnnoise_conv_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cnnoise/conv_5/BiasAddBiasAddcnnoise/conv_5/Conv2D:output:0-cnnoise/conv_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
"cnnoise/batchnorm_5/ReadVariableOpReadVariableOp+cnnoise_batchnorm_5_readvariableop_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_5/ReadVariableOp_1ReadVariableOp-cnnoise_batchnorm_5_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp<cnnoise_batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>cnnoise_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_5/FusedBatchNormV3FusedBatchNormV3cnnoise/conv_5/BiasAdd:output:0*cnnoise/batchnorm_5/ReadVariableOp:value:0,cnnoise/batchnorm_5/ReadVariableOp_1:value:0;cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:0=cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������A:::::*
epsilon%o�:*
is_training( �
cnnoise/activation_4/ReluRelu(cnnoise/batchnorm_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������A�
$cnnoise/conv_6/Conv2D/ReadVariableOpReadVariableOp-cnnoise_conv_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
cnnoise/conv_6/Conv2DConv2D'cnnoise/activation_4/Relu:activations:0,cnnoise/conv_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
�
%cnnoise/conv_6/BiasAdd/ReadVariableOpReadVariableOp.cnnoise_conv_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cnnoise/conv_6/BiasAddBiasAddcnnoise/conv_6/Conv2D:output:0-cnnoise/conv_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
"cnnoise/batchnorm_6/ReadVariableOpReadVariableOp+cnnoise_batchnorm_6_readvariableop_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_6/ReadVariableOp_1ReadVariableOp-cnnoise_batchnorm_6_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOp<cnnoise_batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>cnnoise_batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$cnnoise/batchnorm_6/FusedBatchNormV3FusedBatchNormV3cnnoise/conv_6/BiasAdd:output:0*cnnoise/batchnorm_6/ReadVariableOp:value:0,cnnoise/batchnorm_6/ReadVariableOp_1:value:0;cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:0=cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������A:::::*
epsilon%o�:*
is_training( �
cnnoise/activation_5/ReluRelu(cnnoise/batchnorm_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������A�
$cnnoise/conv_7/Conv2D/ReadVariableOpReadVariableOp-cnnoise_conv_7_conv2d_readvariableop_resource*&
_output_shapes
:A*
dtype0�
cnnoise/conv_7/Conv2DConv2D'cnnoise/activation_5/Relu:activations:0,cnnoise/conv_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
�
%cnnoise/conv_7/BiasAdd/ReadVariableOpReadVariableOp.cnnoise_conv_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cnnoise/conv_7/BiasAddBiasAddcnnoise/conv_7/Conv2D:output:0-cnnoise/conv_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
&cnnoise/dense/Tensordot/ReadVariableOpReadVariableOp/cnnoise_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
cnnoise/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
cnnoise/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          z
cnnoise/dense/Tensordot/ShapeShapecnnoise/conv_7/BiasAdd:output:0*
T0*
_output_shapes
::��g
%cnnoise/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 cnnoise/dense/Tensordot/GatherV2GatherV2&cnnoise/dense/Tensordot/Shape:output:0%cnnoise/dense/Tensordot/free:output:0.cnnoise/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'cnnoise/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"cnnoise/dense/Tensordot/GatherV2_1GatherV2&cnnoise/dense/Tensordot/Shape:output:0%cnnoise/dense/Tensordot/axes:output:00cnnoise/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
cnnoise/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
cnnoise/dense/Tensordot/ProdProd)cnnoise/dense/Tensordot/GatherV2:output:0&cnnoise/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: i
cnnoise/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
cnnoise/dense/Tensordot/Prod_1Prod+cnnoise/dense/Tensordot/GatherV2_1:output:0(cnnoise/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#cnnoise/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cnnoise/dense/Tensordot/concatConcatV2%cnnoise/dense/Tensordot/free:output:0%cnnoise/dense/Tensordot/axes:output:0,cnnoise/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
cnnoise/dense/Tensordot/stackPack%cnnoise/dense/Tensordot/Prod:output:0'cnnoise/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
!cnnoise/dense/Tensordot/transpose	Transposecnnoise/conv_7/BiasAdd:output:0'cnnoise/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:���������A�
cnnoise/dense/Tensordot/ReshapeReshape%cnnoise/dense/Tensordot/transpose:y:0&cnnoise/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
cnnoise/dense/Tensordot/MatMulMatMul(cnnoise/dense/Tensordot/Reshape:output:0.cnnoise/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
cnnoise/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%cnnoise/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 cnnoise/dense/Tensordot/concat_1ConcatV2)cnnoise/dense/Tensordot/GatherV2:output:0(cnnoise/dense/Tensordot/Const_2:output:0.cnnoise/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
cnnoise/dense/TensordotReshape(cnnoise/dense/Tensordot/MatMul:product:0)cnnoise/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:���������A�
$cnnoise/dense/BiasAdd/ReadVariableOpReadVariableOp-cnnoise_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cnnoise/dense/BiasAddBiasAdd cnnoise/dense/Tensordot:output:0,cnnoise/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Az
cnnoise/dense/SigmoidSigmoidcnnoise/dense/BiasAdd:output:0*
T0*/
_output_shapes
:���������Ap
IdentityIdentitycnnoise/dense/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp4^cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp6^cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1#^cnnoise/batchnorm_1/ReadVariableOp%^cnnoise/batchnorm_1/ReadVariableOp_14^cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp6^cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1#^cnnoise/batchnorm_2/ReadVariableOp%^cnnoise/batchnorm_2/ReadVariableOp_14^cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp6^cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1#^cnnoise/batchnorm_3/ReadVariableOp%^cnnoise/batchnorm_3/ReadVariableOp_14^cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp6^cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1#^cnnoise/batchnorm_4/ReadVariableOp%^cnnoise/batchnorm_4/ReadVariableOp_14^cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp6^cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1#^cnnoise/batchnorm_5/ReadVariableOp%^cnnoise/batchnorm_5/ReadVariableOp_14^cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp6^cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1#^cnnoise/batchnorm_6/ReadVariableOp%^cnnoise/batchnorm_6/ReadVariableOp_1&^cnnoise/conv_1/BiasAdd/ReadVariableOp%^cnnoise/conv_1/Conv2D/ReadVariableOp&^cnnoise/conv_2/BiasAdd/ReadVariableOp%^cnnoise/conv_2/Conv2D/ReadVariableOp&^cnnoise/conv_3/BiasAdd/ReadVariableOp%^cnnoise/conv_3/Conv2D/ReadVariableOp&^cnnoise/conv_4/BiasAdd/ReadVariableOp%^cnnoise/conv_4/Conv2D/ReadVariableOp&^cnnoise/conv_5/BiasAdd/ReadVariableOp%^cnnoise/conv_5/Conv2D/ReadVariableOp&^cnnoise/conv_6/BiasAdd/ReadVariableOp%^cnnoise/conv_6/Conv2D/ReadVariableOp&^cnnoise/conv_7/BiasAdd/ReadVariableOp%^cnnoise/conv_7/Conv2D/ReadVariableOp%^cnnoise/dense/BiasAdd/ReadVariableOp'^cnnoise/dense/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������A: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp3cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp2n
5cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp_15cnnoise/batchnorm_1/FusedBatchNormV3/ReadVariableOp_12H
"cnnoise/batchnorm_1/ReadVariableOp"cnnoise/batchnorm_1/ReadVariableOp2L
$cnnoise/batchnorm_1/ReadVariableOp_1$cnnoise/batchnorm_1/ReadVariableOp_12j
3cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp3cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp2n
5cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp_15cnnoise/batchnorm_2/FusedBatchNormV3/ReadVariableOp_12H
"cnnoise/batchnorm_2/ReadVariableOp"cnnoise/batchnorm_2/ReadVariableOp2L
$cnnoise/batchnorm_2/ReadVariableOp_1$cnnoise/batchnorm_2/ReadVariableOp_12j
3cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp3cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp2n
5cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp_15cnnoise/batchnorm_3/FusedBatchNormV3/ReadVariableOp_12H
"cnnoise/batchnorm_3/ReadVariableOp"cnnoise/batchnorm_3/ReadVariableOp2L
$cnnoise/batchnorm_3/ReadVariableOp_1$cnnoise/batchnorm_3/ReadVariableOp_12j
3cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp3cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp2n
5cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp_15cnnoise/batchnorm_4/FusedBatchNormV3/ReadVariableOp_12H
"cnnoise/batchnorm_4/ReadVariableOp"cnnoise/batchnorm_4/ReadVariableOp2L
$cnnoise/batchnorm_4/ReadVariableOp_1$cnnoise/batchnorm_4/ReadVariableOp_12j
3cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp3cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp2n
5cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp_15cnnoise/batchnorm_5/FusedBatchNormV3/ReadVariableOp_12H
"cnnoise/batchnorm_5/ReadVariableOp"cnnoise/batchnorm_5/ReadVariableOp2L
$cnnoise/batchnorm_5/ReadVariableOp_1$cnnoise/batchnorm_5/ReadVariableOp_12j
3cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp3cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp2n
5cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp_15cnnoise/batchnorm_6/FusedBatchNormV3/ReadVariableOp_12H
"cnnoise/batchnorm_6/ReadVariableOp"cnnoise/batchnorm_6/ReadVariableOp2L
$cnnoise/batchnorm_6/ReadVariableOp_1$cnnoise/batchnorm_6/ReadVariableOp_12N
%cnnoise/conv_1/BiasAdd/ReadVariableOp%cnnoise/conv_1/BiasAdd/ReadVariableOp2L
$cnnoise/conv_1/Conv2D/ReadVariableOp$cnnoise/conv_1/Conv2D/ReadVariableOp2N
%cnnoise/conv_2/BiasAdd/ReadVariableOp%cnnoise/conv_2/BiasAdd/ReadVariableOp2L
$cnnoise/conv_2/Conv2D/ReadVariableOp$cnnoise/conv_2/Conv2D/ReadVariableOp2N
%cnnoise/conv_3/BiasAdd/ReadVariableOp%cnnoise/conv_3/BiasAdd/ReadVariableOp2L
$cnnoise/conv_3/Conv2D/ReadVariableOp$cnnoise/conv_3/Conv2D/ReadVariableOp2N
%cnnoise/conv_4/BiasAdd/ReadVariableOp%cnnoise/conv_4/BiasAdd/ReadVariableOp2L
$cnnoise/conv_4/Conv2D/ReadVariableOp$cnnoise/conv_4/Conv2D/ReadVariableOp2N
%cnnoise/conv_5/BiasAdd/ReadVariableOp%cnnoise/conv_5/BiasAdd/ReadVariableOp2L
$cnnoise/conv_5/Conv2D/ReadVariableOp$cnnoise/conv_5/Conv2D/ReadVariableOp2N
%cnnoise/conv_6/BiasAdd/ReadVariableOp%cnnoise/conv_6/BiasAdd/ReadVariableOp2L
$cnnoise/conv_6/Conv2D/ReadVariableOp$cnnoise/conv_6/Conv2D/ReadVariableOp2N
%cnnoise/conv_7/BiasAdd/ReadVariableOp%cnnoise/conv_7/BiasAdd/ReadVariableOp2L
$cnnoise/conv_7/Conv2D/ReadVariableOp$cnnoise/conv_7/Conv2D/ReadVariableOp2L
$cnnoise/dense/BiasAdd/ReadVariableOp$cnnoise/dense/BiasAdd/ReadVariableOp2P
&cnnoise/dense/Tensordot/ReadVariableOp&cnnoise/dense/Tensordot/ReadVariableOp:] Y
/
_output_shapes
:���������A
&
_user_specified_nameconv_1_input:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource
�
�
__inference_loss_fn_9_710840012D
6conv_5_bias_regularizer_l2loss_readvariableop_resource:
identity��-conv_5/bias/Regularizer/L2Loss/ReadVariableOp�
-conv_5/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6conv_5_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_5/bias/Regularizer/L2LossL2Loss5conv_5/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_5/bias/Regularizer/mulMul&conv_5/bias/Regularizer/mul/x:output:0'conv_5/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_5/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^conv_5/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-conv_5/bias/Regularizer/L2Loss/ReadVariableOp-conv_5/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
*__inference_conv_7_layer_call_fn_710839874

inputs!
unknown:A
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_7_layer_call_and_return_conditional_losses_710838540w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839868:)%
#
_user_specified_name	710839870
�

�
/__inference_batchnorm_5_layer_call_fn_710839707

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710838186�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839697:)%
#
_user_specified_name	710839699:)%
#
_user_specified_name	710839701:)%
#
_user_specified_name	710839703
�
L
0__inference_activation_2_layer_call_fn_710839563

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_2_layer_call_and_return_conditional_losses_710838407h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_710839964D
6conv_2_bias_regularizer_l2loss_readvariableop_resource:
identity��-conv_2/bias/Regularizer/L2Loss/ReadVariableOp�
-conv_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6conv_2_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_2/bias/Regularizer/L2LossL2Loss5conv_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0'conv_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_2/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^conv_2/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-conv_2/bias/Regularizer/L2Loss/ReadVariableOp-conv_2/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
*__inference_conv_4_layer_call_fn_710839577

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_4_layer_call_and_return_conditional_losses_710838426w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839571:)%
#
_user_specified_name	710839573
��
�
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838639
conv_1_input*
conv_1_710838313:
conv_1_710838315:#
batchnorm_1_710838318:#
batchnorm_1_710838320:#
batchnorm_1_710838322:#
batchnorm_1_710838324:*
conv_2_710838351:
conv_2_710838353:#
batchnorm_2_710838356:#
batchnorm_2_710838358:#
batchnorm_2_710838360:#
batchnorm_2_710838362:*
conv_3_710838389:
conv_3_710838391:#
batchnorm_3_710838394:#
batchnorm_3_710838396:#
batchnorm_3_710838398:#
batchnorm_3_710838400:*
conv_4_710838427:
conv_4_710838429:#
batchnorm_4_710838432:#
batchnorm_4_710838434:#
batchnorm_4_710838436:#
batchnorm_4_710838438:*
conv_5_710838465:
conv_5_710838467:#
batchnorm_5_710838470:#
batchnorm_5_710838472:#
batchnorm_5_710838474:#
batchnorm_5_710838476:*
conv_6_710838503:
conv_6_710838505:#
batchnorm_6_710838508:#
batchnorm_6_710838510:#
batchnorm_6_710838512:#
batchnorm_6_710838514:*
conv_7_710838541:A
conv_7_710838543:!
dense_710838577:
dense_710838579:
identity��#batchnorm_1/StatefulPartitionedCall�#batchnorm_2/StatefulPartitionedCall�#batchnorm_3/StatefulPartitionedCall�#batchnorm_4/StatefulPartitionedCall�#batchnorm_5/StatefulPartitionedCall�#batchnorm_6/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�-conv_1/bias/Regularizer/L2Loss/ReadVariableOp�/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp�conv_2/StatefulPartitionedCall�-conv_2/bias/Regularizer/L2Loss/ReadVariableOp�/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp�conv_3/StatefulPartitionedCall�-conv_3/bias/Regularizer/L2Loss/ReadVariableOp�/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp�conv_4/StatefulPartitionedCall�-conv_4/bias/Regularizer/L2Loss/ReadVariableOp�/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp�conv_5/StatefulPartitionedCall�-conv_5/bias/Regularizer/L2Loss/ReadVariableOp�/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp�conv_6/StatefulPartitionedCall�-conv_6/bias/Regularizer/L2Loss/ReadVariableOp�/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp�conv_7/StatefulPartitionedCall�-conv_7/bias/Regularizer/L2Loss/ReadVariableOp�/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp�dense/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCallconv_1_inputconv_1_710838313conv_1_710838315*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_1_layer_call_and_return_conditional_losses_710838312�
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_710838318batchnorm_1_710838320batchnorm_1_710838322batchnorm_1_710838324*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710837938�
activation/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_710838331�
conv_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv_2_710838351conv_2_710838353*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_2_layer_call_and_return_conditional_losses_710838350�
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_710838356batchnorm_2_710838358batchnorm_2_710838360batchnorm_2_710838362*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710838000�
activation_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_710838369�
conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv_3_710838389conv_3_710838391*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_3_layer_call_and_return_conditional_losses_710838388�
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_710838394batchnorm_3_710838396batchnorm_3_710838398batchnorm_3_710838400*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710838062�
activation_2/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_2_layer_call_and_return_conditional_losses_710838407�
conv_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv_4_710838427conv_4_710838429*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_4_layer_call_and_return_conditional_losses_710838426�
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batchnorm_4_710838432batchnorm_4_710838434batchnorm_4_710838436batchnorm_4_710838438*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710838124�
activation_3/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_3_layer_call_and_return_conditional_losses_710838445�
conv_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv_5_710838465conv_5_710838467*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_5_layer_call_and_return_conditional_losses_710838464�
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batchnorm_5_710838470batchnorm_5_710838472batchnorm_5_710838474batchnorm_5_710838476*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710838186�
activation_4/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_4_layer_call_and_return_conditional_losses_710838483�
conv_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv_6_710838503conv_6_710838505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_6_layer_call_and_return_conditional_losses_710838502�
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batchnorm_6_710838508batchnorm_6_710838510batchnorm_6_710838512batchnorm_6_710838514*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710838248�
activation_5/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_activation_5_layer_call_and_return_conditional_losses_710838521�
conv_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv_7_710838541conv_7_710838543*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv_7_layer_call_and_return_conditional_losses_710838540�
dense/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0dense_710838577dense_710838579*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_710838576�
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_1_710838313*&
_output_shapes
:*
dtype0�
 conv_1/kernel/Regularizer/L2LossL2Loss7conv_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0)conv_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_1_710838315*
_output_shapes
:*
dtype0�
conv_1/bias/Regularizer/L2LossL2Loss5conv_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0'conv_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_2_710838351*&
_output_shapes
:*
dtype0�
 conv_2/kernel/Regularizer/L2LossL2Loss7conv_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0)conv_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_2_710838353*
_output_shapes
:*
dtype0�
conv_2/bias/Regularizer/L2LossL2Loss5conv_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0'conv_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_3_710838389*&
_output_shapes
:*
dtype0�
 conv_3/kernel/Regularizer/L2LossL2Loss7conv_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0)conv_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_3/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_3_710838391*
_output_shapes
:*
dtype0�
conv_3/bias/Regularizer/L2LossL2Loss5conv_3/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0'conv_3/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_4_710838427*&
_output_shapes
:*
dtype0�
 conv_4/kernel/Regularizer/L2LossL2Loss7conv_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_4/kernel/Regularizer/mulMul(conv_4/kernel/Regularizer/mul/x:output:0)conv_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_4/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_4_710838429*
_output_shapes
:*
dtype0�
conv_4/bias/Regularizer/L2LossL2Loss5conv_4/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_4/bias/Regularizer/mulMul&conv_4/bias/Regularizer/mul/x:output:0'conv_4/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_5_710838465*&
_output_shapes
:*
dtype0�
 conv_5/kernel/Regularizer/L2LossL2Loss7conv_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_5/kernel/Regularizer/mulMul(conv_5/kernel/Regularizer/mul/x:output:0)conv_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_5/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_5_710838467*
_output_shapes
:*
dtype0�
conv_5/bias/Regularizer/L2LossL2Loss5conv_5/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_5/bias/Regularizer/mulMul&conv_5/bias/Regularizer/mul/x:output:0'conv_5/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_6_710838503*&
_output_shapes
:*
dtype0�
 conv_6/kernel/Regularizer/L2LossL2Loss7conv_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_6/kernel/Regularizer/mulMul(conv_6/kernel/Regularizer/mul/x:output:0)conv_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_6/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_6_710838505*
_output_shapes
:*
dtype0�
conv_6/bias/Regularizer/L2LossL2Loss5conv_6/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_6/bias/Regularizer/mulMul&conv_6/bias/Regularizer/mul/x:output:0'conv_6/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_7_710838541*&
_output_shapes
:A*
dtype0�
 conv_7/kernel/Regularizer/L2LossL2Loss7conv_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_7/kernel/Regularizer/mulMul(conv_7/kernel/Regularizer/mul/x:output:0)conv_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
-conv_7/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_7_710838543*
_output_shapes
:*
dtype0�
conv_7/bias/Regularizer/L2LossL2Loss5conv_7/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_7/bias/Regularizer/mulMul&conv_7/bias/Regularizer/mul/x:output:0'conv_7/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: }
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A�	
NoOpNoOp$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall^conv_1/StatefulPartitionedCall.^conv_1/bias/Regularizer/L2Loss/ReadVariableOp0^conv_1/kernel/Regularizer/L2Loss/ReadVariableOp^conv_2/StatefulPartitionedCall.^conv_2/bias/Regularizer/L2Loss/ReadVariableOp0^conv_2/kernel/Regularizer/L2Loss/ReadVariableOp^conv_3/StatefulPartitionedCall.^conv_3/bias/Regularizer/L2Loss/ReadVariableOp0^conv_3/kernel/Regularizer/L2Loss/ReadVariableOp^conv_4/StatefulPartitionedCall.^conv_4/bias/Regularizer/L2Loss/ReadVariableOp0^conv_4/kernel/Regularizer/L2Loss/ReadVariableOp^conv_5/StatefulPartitionedCall.^conv_5/bias/Regularizer/L2Loss/ReadVariableOp0^conv_5/kernel/Regularizer/L2Loss/ReadVariableOp^conv_6/StatefulPartitionedCall.^conv_6/bias/Regularizer/L2Loss/ReadVariableOp0^conv_6/kernel/Regularizer/L2Loss/ReadVariableOp^conv_7/StatefulPartitionedCall.^conv_7/bias/Regularizer/L2Loss/ReadVariableOp0^conv_7/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������A: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2^
-conv_1/bias/Regularizer/L2Loss/ReadVariableOp-conv_1/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2^
-conv_2/bias/Regularizer/L2Loss/ReadVariableOp-conv_2/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp/conv_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2^
-conv_3/bias/Regularizer/L2Loss/ReadVariableOp-conv_3/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2^
-conv_4/bias/Regularizer/L2Loss/ReadVariableOp-conv_4/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2^
-conv_5/bias/Regularizer/L2Loss/ReadVariableOp-conv_5/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2^
-conv_6/bias/Regularizer/L2Loss/ReadVariableOp-conv_6/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2^
-conv_7/bias/Regularizer/L2Loss/ReadVariableOp-conv_7/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
/
_output_shapes
:���������A
&
_user_specified_nameconv_1_input:)%
#
_user_specified_name	710838313:)%
#
_user_specified_name	710838315:)%
#
_user_specified_name	710838318:)%
#
_user_specified_name	710838320:)%
#
_user_specified_name	710838322:)%
#
_user_specified_name	710838324:)%
#
_user_specified_name	710838351:)%
#
_user_specified_name	710838353:)	%
#
_user_specified_name	710838356:)
%
#
_user_specified_name	710838358:)%
#
_user_specified_name	710838360:)%
#
_user_specified_name	710838362:)%
#
_user_specified_name	710838389:)%
#
_user_specified_name	710838391:)%
#
_user_specified_name	710838394:)%
#
_user_specified_name	710838396:)%
#
_user_specified_name	710838398:)%
#
_user_specified_name	710838400:)%
#
_user_specified_name	710838427:)%
#
_user_specified_name	710838429:)%
#
_user_specified_name	710838432:)%
#
_user_specified_name	710838434:)%
#
_user_specified_name	710838436:)%
#
_user_specified_name	710838438:)%
#
_user_specified_name	710838465:)%
#
_user_specified_name	710838467:)%
#
_user_specified_name	710838470:)%
#
_user_specified_name	710838472:)%
#
_user_specified_name	710838474:)%
#
_user_specified_name	710838476:)%
#
_user_specified_name	710838503:) %
#
_user_specified_name	710838505:)!%
#
_user_specified_name	710838508:)"%
#
_user_specified_name	710838510:)#%
#
_user_specified_name	710838512:)$%
#
_user_specified_name	710838514:)%%
#
_user_specified_name	710838541:)&%
#
_user_specified_name	710838543:)'%
#
_user_specified_name	710838577:)(%
#
_user_specified_name	710838579
�
Y
%__inference__update_step_xla_14226618
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
D__inference_dense_layer_call_and_return_conditional_losses_710838576

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:���������A�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:���������Ar
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������Ab
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������AV
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
M
%__inference__update_step_xla_14226608
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
g
K__inference_activation_1_layer_call_and_return_conditional_losses_710839469

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
�
E__inference_conv_5_layer_call_and_return_conditional_losses_710838464

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_5/bias/Regularizer/L2Loss/ReadVariableOp�/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_5/kernel/Regularizer/L2LossL2Loss7conv_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_5/kernel/Regularizer/mulMul(conv_5/kernel/Regularizer/mul/x:output:0)conv_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_5/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_5/bias/Regularizer/L2LossL2Loss5conv_5/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_5/bias/Regularizer/mulMul&conv_5/bias/Regularizer/mul/x:output:0'conv_5/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_5/bias/Regularizer/L2Loss/ReadVariableOp0^conv_5/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_5/bias/Regularizer/L2Loss/ReadVariableOp-conv_5/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp/conv_5/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
g
K__inference_activation_1_layer_call_and_return_conditional_losses_710838369

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�	
�
 __inference_loss_fn_12_710840036R
8conv_7_kernel_regularizer_l2loss_readvariableop_resource:A
identity��/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp�
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv_7_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:A*
dtype0�
 conv_7/kernel/Regularizer/L2LossL2Loss7conv_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_7/kernel/Regularizer/mulMul(conv_7/kernel/Regularizer/mul/x:output:0)conv_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_7/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: T
NoOpNoOp0^conv_7/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp/conv_7/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710838062

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
Y
%__inference__update_step_xla_14226598
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710838000

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv_3_layer_call_and_return_conditional_losses_710838388

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_3/bias/Regularizer/L2Loss/ReadVariableOp�/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_3/kernel/Regularizer/L2LossL2Loss7conv_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0)conv_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_3/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_3/bias/Regularizer/L2LossL2Loss5conv_3/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0'conv_3/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_3/bias/Regularizer/L2Loss/ReadVariableOp0^conv_3/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_3/bias/Regularizer/L2Loss/ReadVariableOp-conv_3/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp/conv_3/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
__inference_loss_fn_1_710839948D
6conv_1_bias_regularizer_l2loss_readvariableop_resource:
identity��-conv_1/bias/Regularizer/L2Loss/ReadVariableOp�
-conv_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6conv_1_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_1/bias/Regularizer/L2LossL2Loss5conv_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0'conv_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^conv_1/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-conv_1/bias/Regularizer/L2Loss/ReadVariableOp-conv_1/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�"
�	
+__inference_cnnoise_layer_call_fn_710838884
conv_1_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:A

unknown_36:

unknown_37:

unknown_38:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*>
_read_only_resource_inputs 
	
 !"%&'(*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838639w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������A: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������A
&
_user_specified_nameconv_1_input:)%
#
_user_specified_name	710838802:)%
#
_user_specified_name	710838804:)%
#
_user_specified_name	710838806:)%
#
_user_specified_name	710838808:)%
#
_user_specified_name	710838810:)%
#
_user_specified_name	710838812:)%
#
_user_specified_name	710838814:)%
#
_user_specified_name	710838816:)	%
#
_user_specified_name	710838818:)
%
#
_user_specified_name	710838820:)%
#
_user_specified_name	710838822:)%
#
_user_specified_name	710838824:)%
#
_user_specified_name	710838826:)%
#
_user_specified_name	710838828:)%
#
_user_specified_name	710838830:)%
#
_user_specified_name	710838832:)%
#
_user_specified_name	710838834:)%
#
_user_specified_name	710838836:)%
#
_user_specified_name	710838838:)%
#
_user_specified_name	710838840:)%
#
_user_specified_name	710838842:)%
#
_user_specified_name	710838844:)%
#
_user_specified_name	710838846:)%
#
_user_specified_name	710838848:)%
#
_user_specified_name	710838850:)%
#
_user_specified_name	710838852:)%
#
_user_specified_name	710838854:)%
#
_user_specified_name	710838856:)%
#
_user_specified_name	710838858:)%
#
_user_specified_name	710838860:)%
#
_user_specified_name	710838862:) %
#
_user_specified_name	710838864:)!%
#
_user_specified_name	710838866:)"%
#
_user_specified_name	710838868:)#%
#
_user_specified_name	710838870:)$%
#
_user_specified_name	710838872:)%%
#
_user_specified_name	710838874:)&%
#
_user_specified_name	710838876:)'%
#
_user_specified_name	710838878:)(%
#
_user_specified_name	710838880
�

�
/__inference_batchnorm_4_layer_call_fn_710839608

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710838124�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839598:)%
#
_user_specified_name	710839600:)%
#
_user_specified_name	710839602:)%
#
_user_specified_name	710839604
�
�
 __inference_loss_fn_13_710840044D
6conv_7_bias_regularizer_l2loss_readvariableop_resource:
identity��-conv_7/bias/Regularizer/L2Loss/ReadVariableOp�
-conv_7/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6conv_7_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_7/bias/Regularizer/L2LossL2Loss5conv_7/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_7/bias/Regularizer/mulMul&conv_7/bias/Regularizer/mul/x:output:0'conv_7/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_7/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^conv_7/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-conv_7/bias/Regularizer/L2Loss/ReadVariableOp-conv_7/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�	
�
__inference_loss_fn_6_710839988R
8conv_4_kernel_regularizer_l2loss_readvariableop_resource:
identity��/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp�
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv_4_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_4/kernel/Regularizer/L2LossL2Loss7conv_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_4/kernel/Regularizer/mulMul(conv_4/kernel/Regularizer/mul/x:output:0)conv_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: T
NoOpNoOp0^conv_4/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp/conv_4/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
M
%__inference__update_step_xla_14226663
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
M
%__inference__update_step_xla_14226603
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
)__inference_dense_layer_call_fn_710839901

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������A*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_710838576w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������A<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839895:)%
#
_user_specified_name	710839897
�
�
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710839657

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
g
K__inference_activation_4_layer_call_and_return_conditional_losses_710839766

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
�
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710838266

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
/__inference_batchnorm_6_layer_call_fn_710839819

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710838266�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839809:)%
#
_user_specified_name	710839811:)%
#
_user_specified_name	710839813:)%
#
_user_specified_name	710839815
�
g
K__inference_activation_3_layer_call_and_return_conditional_losses_710838445

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�
M
%__inference__update_step_xla_14226633
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
M
%__inference__update_step_xla_14226568
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
/__inference_batchnorm_3_layer_call_fn_710839509

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710838062�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:)%
#
_user_specified_name	710839499:)%
#
_user_specified_name	710839501:)%
#
_user_specified_name	710839503:)%
#
_user_specified_name	710839505
�
e
I__inference_activation_layer_call_and_return_conditional_losses_710839370

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������Ab
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������A:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_710839940R
8conv_1_kernel_regularizer_l2loss_readvariableop_resource:
identity��/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp�
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv_1_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_1/kernel/Regularizer/L2LossL2Loss7conv_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0)conv_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: T
NoOpNoOp0^conv_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp/conv_1/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
E__inference_conv_6_layer_call_and_return_conditional_losses_710838502

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�-conv_6/bias/Regularizer/L2Loss/ReadVariableOp�/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������A�
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv_6/kernel/Regularizer/L2LossL2Loss7conv_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
conv_6/kernel/Regularizer/mulMul(conv_6/kernel/Regularizer/mul/x:output:0)conv_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-conv_6/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_6/bias/Regularizer/L2LossL2Loss5conv_6/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_6/bias/Regularizer/mulMul&conv_6/bias/Regularizer/mul/x:output:0'conv_6/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������A�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp.^conv_6/bias/Regularizer/L2Loss/ReadVariableOp0^conv_6/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2^
-conv_6/bias/Regularizer/L2Loss/ReadVariableOp-conv_6/bias/Regularizer/L2Loss/ReadVariableOp2b
/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp/conv_6/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������A
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
M
%__inference__update_step_xla_14226613
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710839756

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
__inference_loss_fn_7_710839996D
6conv_4_bias_regularizer_l2loss_readvariableop_resource:
identity��-conv_4/bias/Regularizer/L2Loss/ReadVariableOp�
-conv_4/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6conv_4_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_4/bias/Regularizer/L2LossL2Loss5conv_4/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
conv_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
conv_4/bias/Regularizer/mulMul&conv_4/bias/Regularizer/mul/x:output:0'conv_4/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_4/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^conv_4/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-conv_4/bias/Regularizer/L2Loss/ReadVariableOp-conv_4/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
conv_1_input=
serving_default_conv_1_input:0���������AA
dense8
StatefulPartitionedCall:0���������Atensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
test_xspectra
test_features
test_segments
test_audio_list
 dirs

!params
"params_feat
#	optimizer
$
signatures"
_tf_keras_sequential
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4axis
	5gamma
6beta
7moving_mean
8moving_variance"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
 G_jit_compiled_convolution_op"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

_kernel
`bias
 a_jit_compiled_convolution_op"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias
 {_jit_compiled_convolution_op"
_tf_keras_layer
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
+0
,1
52
63
74
85
E6
F7
O8
P9
Q10
R11
_12
`13
i14
j15
k16
l17
y18
z19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39"
trackable_list_wrapper
�
+0
,1
52
63
E4
F5
O6
P7
_8
`9
i10
j11
y12
z13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_cnnoise_layer_call_fn_710838884
+__inference_cnnoise_layer_call_fn_710838969�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838639
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838799�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
$__inference__wrapped_model_710837920conv_1_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv_1_layer_call_fn_710839280�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv_1_layer_call_and_return_conditional_losses_710839298�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv_1/kernel
:2conv_1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_batchnorm_1_layer_call_fn_710839311
/__inference_batchnorm_1_layer_call_fn_710839324�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710839342
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710839360�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:2batchnorm_1/gamma
:2batchnorm_1/beta
':% (2batchnorm_1/moving_mean
+:) (2batchnorm_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_activation_layer_call_fn_710839365�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_activation_layer_call_and_return_conditional_losses_710839370�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv_2_layer_call_fn_710839379�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv_2_layer_call_and_return_conditional_losses_710839397�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv_2/kernel
:2conv_2/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_batchnorm_2_layer_call_fn_710839410
/__inference_batchnorm_2_layer_call_fn_710839423�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710839441
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710839459�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:2batchnorm_2/gamma
:2batchnorm_2/beta
':% (2batchnorm_2/moving_mean
+:) (2batchnorm_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_activation_1_layer_call_fn_710839464�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_activation_1_layer_call_and_return_conditional_losses_710839469�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv_3_layer_call_fn_710839478�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv_3_layer_call_and_return_conditional_losses_710839496�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv_3/kernel
:2conv_3/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
i0
j1
k2
l3"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_batchnorm_3_layer_call_fn_710839509
/__inference_batchnorm_3_layer_call_fn_710839522�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710839540
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710839558�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:2batchnorm_3/gamma
:2batchnorm_3/beta
':% (2batchnorm_3/moving_mean
+:) (2batchnorm_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_activation_2_layer_call_fn_710839563�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_activation_2_layer_call_and_return_conditional_losses_710839568�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv_4_layer_call_fn_710839577�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv_4_layer_call_and_return_conditional_losses_710839595�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv_4/kernel
:2conv_4/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_batchnorm_4_layer_call_fn_710839608
/__inference_batchnorm_4_layer_call_fn_710839621�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710839639
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710839657�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:2batchnorm_4/gamma
:2batchnorm_4/beta
':% (2batchnorm_4/moving_mean
+:) (2batchnorm_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_activation_3_layer_call_fn_710839662�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_activation_3_layer_call_and_return_conditional_losses_710839667�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv_5_layer_call_fn_710839676�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv_5_layer_call_and_return_conditional_losses_710839694�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv_5/kernel
:2conv_5/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_batchnorm_5_layer_call_fn_710839707
/__inference_batchnorm_5_layer_call_fn_710839720�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710839738
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710839756�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:2batchnorm_5/gamma
:2batchnorm_5/beta
':% (2batchnorm_5/moving_mean
+:) (2batchnorm_5/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_activation_4_layer_call_fn_710839761�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_activation_4_layer_call_and_return_conditional_losses_710839766�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv_6_layer_call_fn_710839775�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv_6_layer_call_and_return_conditional_losses_710839793�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv_6/kernel
:2conv_6/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_batchnorm_6_layer_call_fn_710839806
/__inference_batchnorm_6_layer_call_fn_710839819�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710839837
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710839855�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:2batchnorm_6/gamma
:2batchnorm_6/beta
':% (2batchnorm_6/moving_mean
+:) (2batchnorm_6/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_activation_5_layer_call_fn_710839860�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_activation_5_layer_call_and_return_conditional_losses_710839865�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv_7_layer_call_fn_710839874�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv_7_layer_call_and_return_conditional_losses_710839892�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%A2conv_7/kernel
:2conv_7/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_layer_call_fn_710839901�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_layer_call_and_return_conditional_losses_710839932�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:2dense/kernel
:2
dense/bias
�
�trace_02�
__inference_loss_fn_0_710839940�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_710839948�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_710839956�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_710839964�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_710839972�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_710839980�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_710839988�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_710839996�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_8_710840004�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_9_710840012�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
 __inference_loss_fn_10_710840020�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
 __inference_loss_fn_11_710840028�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
 __inference_loss_fn_12_710840036�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
 __inference_loss_fn_13_710840044�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
|
70
81
Q2
R3
k4
l5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_cnnoise_layer_call_fn_710838884conv_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_cnnoise_layer_call_fn_710838969conv_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838639conv_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838799conv_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15
�trace_16
�trace_17
�trace_18
�trace_19
�trace_20
�trace_21
�trace_22
�trace_23
�trace_24
�trace_25
�trace_26
�trace_272�	
%__inference__update_step_xla_14226538
%__inference__update_step_xla_14226543
%__inference__update_step_xla_14226548
%__inference__update_step_xla_14226553
%__inference__update_step_xla_14226558
%__inference__update_step_xla_14226563
%__inference__update_step_xla_14226568
%__inference__update_step_xla_14226573
%__inference__update_step_xla_14226578
%__inference__update_step_xla_14226583
%__inference__update_step_xla_14226588
%__inference__update_step_xla_14226593
%__inference__update_step_xla_14226598
%__inference__update_step_xla_14226603
%__inference__update_step_xla_14226608
%__inference__update_step_xla_14226613
%__inference__update_step_xla_14226618
%__inference__update_step_xla_14226623
%__inference__update_step_xla_14226628
%__inference__update_step_xla_14226633
%__inference__update_step_xla_14226638
%__inference__update_step_xla_14226643
%__inference__update_step_xla_14226648
%__inference__update_step_xla_14226653
%__inference__update_step_xla_14226658
%__inference__update_step_xla_14226663
%__inference__update_step_xla_14226668
%__inference__update_step_xla_14226673�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11z�trace_12z�trace_13z�trace_14z�trace_15z�trace_16z�trace_17z�trace_18z�trace_19z�trace_20z�trace_21z�trace_22z�trace_23z�trace_24z�trace_25z�trace_26z�trace_27
�B�
'__inference_signature_wrapper_710839215conv_1_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv_1_layer_call_fn_710839280inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv_1_layer_call_and_return_conditional_losses_710839298inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_batchnorm_1_layer_call_fn_710839311inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_batchnorm_1_layer_call_fn_710839324inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710839342inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710839360inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_activation_layer_call_fn_710839365inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_activation_layer_call_and_return_conditional_losses_710839370inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv_2_layer_call_fn_710839379inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv_2_layer_call_and_return_conditional_losses_710839397inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_batchnorm_2_layer_call_fn_710839410inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_batchnorm_2_layer_call_fn_710839423inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710839441inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710839459inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_activation_1_layer_call_fn_710839464inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_activation_1_layer_call_and_return_conditional_losses_710839469inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv_3_layer_call_fn_710839478inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv_3_layer_call_and_return_conditional_losses_710839496inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_batchnorm_3_layer_call_fn_710839509inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_batchnorm_3_layer_call_fn_710839522inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710839540inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710839558inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_activation_2_layer_call_fn_710839563inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_activation_2_layer_call_and_return_conditional_losses_710839568inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv_4_layer_call_fn_710839577inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv_4_layer_call_and_return_conditional_losses_710839595inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_batchnorm_4_layer_call_fn_710839608inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_batchnorm_4_layer_call_fn_710839621inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710839639inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710839657inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_activation_3_layer_call_fn_710839662inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_activation_3_layer_call_and_return_conditional_losses_710839667inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv_5_layer_call_fn_710839676inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv_5_layer_call_and_return_conditional_losses_710839694inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_batchnorm_5_layer_call_fn_710839707inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_batchnorm_5_layer_call_fn_710839720inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710839738inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710839756inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_activation_4_layer_call_fn_710839761inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_activation_4_layer_call_and_return_conditional_losses_710839766inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv_6_layer_call_fn_710839775inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv_6_layer_call_and_return_conditional_losses_710839793inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_batchnorm_6_layer_call_fn_710839806inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_batchnorm_6_layer_call_fn_710839819inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710839837inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710839855inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_activation_5_layer_call_fn_710839860inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_activation_5_layer_call_and_return_conditional_losses_710839865inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv_7_layer_call_fn_710839874inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv_7_layer_call_and_return_conditional_losses_710839892inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_layer_call_fn_710839901inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_layer_call_and_return_conditional_losses_710839932inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_710839940"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_710839948"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_710839956"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_710839964"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_710839972"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_710839980"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_6_710839988"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_7_710839996"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_8_710840004"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_9_710840012"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
 __inference_loss_fn_10_710840020"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
 __inference_loss_fn_11_710840028"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
 __inference_loss_fn_12_710840036"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
 __inference_loss_fn_13_710840044"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
,:*2Adam/m/conv_1/kernel
,:*2Adam/v/conv_1/kernel
:2Adam/m/conv_1/bias
:2Adam/v/conv_1/bias
$:"2Adam/m/batchnorm_1/gamma
$:"2Adam/v/batchnorm_1/gamma
#:!2Adam/m/batchnorm_1/beta
#:!2Adam/v/batchnorm_1/beta
,:*2Adam/m/conv_2/kernel
,:*2Adam/v/conv_2/kernel
:2Adam/m/conv_2/bias
:2Adam/v/conv_2/bias
$:"2Adam/m/batchnorm_2/gamma
$:"2Adam/v/batchnorm_2/gamma
#:!2Adam/m/batchnorm_2/beta
#:!2Adam/v/batchnorm_2/beta
,:*2Adam/m/conv_3/kernel
,:*2Adam/v/conv_3/kernel
:2Adam/m/conv_3/bias
:2Adam/v/conv_3/bias
$:"2Adam/m/batchnorm_3/gamma
$:"2Adam/v/batchnorm_3/gamma
#:!2Adam/m/batchnorm_3/beta
#:!2Adam/v/batchnorm_3/beta
,:*2Adam/m/conv_4/kernel
,:*2Adam/v/conv_4/kernel
:2Adam/m/conv_4/bias
:2Adam/v/conv_4/bias
$:"2Adam/m/batchnorm_4/gamma
$:"2Adam/v/batchnorm_4/gamma
#:!2Adam/m/batchnorm_4/beta
#:!2Adam/v/batchnorm_4/beta
,:*2Adam/m/conv_5/kernel
,:*2Adam/v/conv_5/kernel
:2Adam/m/conv_5/bias
:2Adam/v/conv_5/bias
$:"2Adam/m/batchnorm_5/gamma
$:"2Adam/v/batchnorm_5/gamma
#:!2Adam/m/batchnorm_5/beta
#:!2Adam/v/batchnorm_5/beta
,:*2Adam/m/conv_6/kernel
,:*2Adam/v/conv_6/kernel
:2Adam/m/conv_6/bias
:2Adam/v/conv_6/bias
$:"2Adam/m/batchnorm_6/gamma
$:"2Adam/v/batchnorm_6/gamma
#:!2Adam/m/batchnorm_6/beta
#:!2Adam/v/batchnorm_6/beta
,:*A2Adam/m/conv_7/kernel
,:*A2Adam/v/conv_7/kernel
:2Adam/m/conv_7/bias
:2Adam/v/conv_7/bias
#:!2Adam/m/dense/kernel
#:!2Adam/v/dense/kernel
:2Adam/m/dense/bias
:2Adam/v/dense/bias
�B�
%__inference__update_step_xla_14226538gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226543gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226548gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226553gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226558gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226563gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226568gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226573gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226578gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226583gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226588gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226593gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226598gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226603gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226608gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226613gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226618gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226623gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226628gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226633gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226638gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226643gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226648gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226653gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226658gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226663gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226668gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__update_step_xla_14226673gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
%__inference__update_step_xla_14226538~x�u
n�k
!�
gradient
<�9	%�"
�
�
p
` VariableSpec 
`�����?
� "
 �
%__inference__update_step_xla_14226543f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�����?
� "
 �
%__inference__update_step_xla_14226548f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`��А��?
� "
 �
%__inference__update_step_xla_14226553f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`��ΐ��?
� "
 �
%__inference__update_step_xla_14226558~x�u
n�k
!�
gradient
<�9	%�"
�
�
p
` VariableSpec 
`��ڇ��?
� "
 �
%__inference__update_step_xla_14226563f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`��ۇ��?
� "
 �
%__inference__update_step_xla_14226568f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`��ڇ��?
� "
 �
%__inference__update_step_xla_14226573f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`��ڇ��?
� "
 �
%__inference__update_step_xla_14226578~x�u
n�k
!�
gradient
<�9	%�"
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226583f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226588f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226593f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226598~x�u
n�k
!�
gradient
<�9	%�"
�
�
p
` VariableSpec 
`��·��?
� "
 �
%__inference__update_step_xla_14226603f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�ϐ��?
� "
 �
%__inference__update_step_xla_14226608f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�ɧ���?
� "
 �
%__inference__update_step_xla_14226613f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226618~x�u
n�k
!�
gradient
<�9	%�"
�
�
p
` VariableSpec 
`࣪���?
� "
 �
%__inference__update_step_xla_14226623f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`६���?
� "
 �
%__inference__update_step_xla_14226628f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�ح���?
� "
 �
%__inference__update_step_xla_14226633f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�ݭ���?
� "
 �
%__inference__update_step_xla_14226638~x�u
n�k
!�
gradient
<�9	%�"
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226643f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�ݰ���?
� "
 �
%__inference__update_step_xla_14226648f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226653f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226658~x�u
n�k
!�
gradientA
<�9	%�"
�A
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226663f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`స���?
� "
 �
%__inference__update_step_xla_14226668nh�e
^�[
�
gradient
4�1	�
�
�
p
` VariableSpec 
`������?
� "
 �
%__inference__update_step_xla_14226673f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
$__inference__wrapped_model_710837920�<+,5678EFOPQR_`ijklyz��������������������=�:
3�0
.�+
conv_1_input���������A
� "5�2
0
dense'�$
dense���������A�
K__inference_activation_1_layer_call_and_return_conditional_losses_710839469o7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
0__inference_activation_1_layer_call_fn_710839464d7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
K__inference_activation_2_layer_call_and_return_conditional_losses_710839568o7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
0__inference_activation_2_layer_call_fn_710839563d7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
K__inference_activation_3_layer_call_and_return_conditional_losses_710839667o7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
0__inference_activation_3_layer_call_fn_710839662d7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
K__inference_activation_4_layer_call_and_return_conditional_losses_710839766o7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
0__inference_activation_4_layer_call_fn_710839761d7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
K__inference_activation_5_layer_call_and_return_conditional_losses_710839865o7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
0__inference_activation_5_layer_call_fn_710839860d7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
I__inference_activation_layer_call_and_return_conditional_losses_710839370o7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
.__inference_activation_layer_call_fn_710839365d7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710839342�5678Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
J__inference_batchnorm_1_layer_call_and_return_conditional_losses_710839360�5678Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
/__inference_batchnorm_1_layer_call_fn_710839311�5678Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
/__inference_batchnorm_1_layer_call_fn_710839324�5678Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710839441�OPQRQ�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
J__inference_batchnorm_2_layer_call_and_return_conditional_losses_710839459�OPQRQ�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
/__inference_batchnorm_2_layer_call_fn_710839410�OPQRQ�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
/__inference_batchnorm_2_layer_call_fn_710839423�OPQRQ�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710839540�ijklQ�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
J__inference_batchnorm_3_layer_call_and_return_conditional_losses_710839558�ijklQ�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
/__inference_batchnorm_3_layer_call_fn_710839509�ijklQ�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
/__inference_batchnorm_3_layer_call_fn_710839522�ijklQ�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710839639�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
J__inference_batchnorm_4_layer_call_and_return_conditional_losses_710839657�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
/__inference_batchnorm_4_layer_call_fn_710839608�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
/__inference_batchnorm_4_layer_call_fn_710839621�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710839738�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
J__inference_batchnorm_5_layer_call_and_return_conditional_losses_710839756�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
/__inference_batchnorm_5_layer_call_fn_710839707�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
/__inference_batchnorm_5_layer_call_fn_710839720�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710839837�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
J__inference_batchnorm_6_layer_call_and_return_conditional_losses_710839855�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
/__inference_batchnorm_6_layer_call_fn_710839806�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
/__inference_batchnorm_6_layer_call_fn_710839819�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838639�<+,5678EFOPQR_`ijklyz��������������������E�B
;�8
.�+
conv_1_input���������A
p

 
� "4�1
*�'
tensor_0���������A
� �
F__inference_cnnoise_layer_call_and_return_conditional_losses_710838799�<+,5678EFOPQR_`ijklyz��������������������E�B
;�8
.�+
conv_1_input���������A
p 

 
� "4�1
*�'
tensor_0���������A
� �
+__inference_cnnoise_layer_call_fn_710838884�<+,5678EFOPQR_`ijklyz��������������������E�B
;�8
.�+
conv_1_input���������A
p

 
� ")�&
unknown���������A�
+__inference_cnnoise_layer_call_fn_710838969�<+,5678EFOPQR_`ijklyz��������������������E�B
;�8
.�+
conv_1_input���������A
p 

 
� ")�&
unknown���������A�
E__inference_conv_1_layer_call_and_return_conditional_losses_710839298s+,7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
*__inference_conv_1_layer_call_fn_710839280h+,7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
E__inference_conv_2_layer_call_and_return_conditional_losses_710839397sEF7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
*__inference_conv_2_layer_call_fn_710839379hEF7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
E__inference_conv_3_layer_call_and_return_conditional_losses_710839496s_`7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
*__inference_conv_3_layer_call_fn_710839478h_`7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
E__inference_conv_4_layer_call_and_return_conditional_losses_710839595syz7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
*__inference_conv_4_layer_call_fn_710839577hyz7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
E__inference_conv_5_layer_call_and_return_conditional_losses_710839694u��7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
*__inference_conv_5_layer_call_fn_710839676j��7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
E__inference_conv_6_layer_call_and_return_conditional_losses_710839793u��7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
*__inference_conv_6_layer_call_fn_710839775j��7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
E__inference_conv_7_layer_call_and_return_conditional_losses_710839892u��7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
*__inference_conv_7_layer_call_fn_710839874j��7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������A�
D__inference_dense_layer_call_and_return_conditional_losses_710839932u��7�4
-�*
(�%
inputs���������A
� "4�1
*�'
tensor_0���������A
� �
)__inference_dense_layer_call_fn_710839901j��7�4
-�*
(�%
inputs���������A
� ")�&
unknown���������AG
__inference_loss_fn_0_710839940$+�

� 
� "�
unknown I
 __inference_loss_fn_10_710840020%��

� 
� "�
unknown I
 __inference_loss_fn_11_710840028%��

� 
� "�
unknown I
 __inference_loss_fn_12_710840036%��

� 
� "�
unknown I
 __inference_loss_fn_13_710840044%��

� 
� "�
unknown G
__inference_loss_fn_1_710839948$,�

� 
� "�
unknown G
__inference_loss_fn_2_710839956$E�

� 
� "�
unknown G
__inference_loss_fn_3_710839964$F�

� 
� "�
unknown G
__inference_loss_fn_4_710839972$_�

� 
� "�
unknown G
__inference_loss_fn_5_710839980$`�

� 
� "�
unknown G
__inference_loss_fn_6_710839988$y�

� 
� "�
unknown G
__inference_loss_fn_7_710839996$z�

� 
� "�
unknown H
__inference_loss_fn_8_710840004%��

� 
� "�
unknown H
__inference_loss_fn_9_710840012%��

� 
� "�
unknown �
'__inference_signature_wrapper_710839215�<+,5678EFOPQR_`ijklyz��������������������M�J
� 
C�@
>
conv_1_input.�+
conv_1_input���������A"5�2
0
dense'�$
dense���������A