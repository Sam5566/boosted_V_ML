��-
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��(
{
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	�*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_16/gamma
�
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_16/beta
�
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:*
dtype0
�
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_16/moving_mean
�
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:*
dtype0
�
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_16/moving_variance
�
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:*
dtype0
�
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
: *
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
: *
dtype0
�
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_17/gamma
�
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_17/beta
�
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_17/moving_mean
�
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_17/moving_variance
�
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*!
shared_nameconv2d_17/kernel
~
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*'
_output_shapes
: �*
dtype0
u
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_17/bias
n
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_18/gamma
�
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_18/beta
�
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_18/moving_mean
�
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_18/moving_variance
�
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_18/kernel

$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_18/bias
n
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_19/gamma
�
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_19/beta
�
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_19/moving_mean
�
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_19/moving_variance
�
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_19/kernel

$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_19/bias
n
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes	
:�*
dtype0
}
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*�* 
shared_namedense_12/kernel
v
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*!
_output_shapes
:��*�*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:�*
dtype0
|
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_13/kernel
u
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel* 
_output_shapes
:
��*
dtype0
s
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
:�*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_14/kernel/m
�
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_16/gamma/m
�
7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/m*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_16/beta/m
�
6Adam/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_16/kernel/m
�
+Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_16/bias/m
{
)Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_17/gamma/m
�
7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_17/beta/m
�
6Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_17/kernel/m
�
+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_17/bias/m
|
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_18/gamma/m
�
7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_18/beta/m
�
6Adam/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_18/kernel/m
�
+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_18/bias/m
|
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_19/gamma/m
�
7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_19/beta/m
�
6Adam/batch_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_19/kernel/m
�
+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_19/bias/m
|
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*�*'
shared_nameAdam/dense_12/kernel/m
�
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*!
_output_shapes
:��*�*
dtype0
�
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_13/kernel/m
�
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_13/bias/m
z
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_14/kernel/v
�
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_16/gamma/v
�
7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/v*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_16/beta/v
�
6Adam/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_16/kernel/v
�
+Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_16/bias/v
{
)Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_17/gamma/v
�
7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_17/beta/v
�
6Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_17/kernel/v
�
+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_17/bias/v
|
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_18/gamma/v
�
7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_18/beta/v
�
6Adam/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_18/kernel/v
�
+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_18/bias/v
|
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_19/gamma/v
�
7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_19/beta/v
�
6Adam/batch_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_19/kernel/v
�
+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_19/bias/v
|
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*�*'
shared_nameAdam/dense_12/kernel/v
�
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*!
_output_shapes
:��*�*
dtype0
�
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_13/kernel/v
�
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_13/bias/v
z
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�

h2ptjl
_output
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer_with_weights-6
layer-8
layer_with_weights-7
layer-9
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
�
"iter

#beta_1

$beta_2
	%decay
&learning_ratem�m�'m�(m�+m�,m�-m�.m�1m�2m�3m�4m�7m�8m�9m�:m�=m�>m�?m�@m�Am�Bm�v�v�'v�(v�+v�,v�-v�.v�1v�2v�3v�4v�7v�8v�9v�:v�=v�>v�?v�@v�Av�Bv�
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
@25
A26
B27
28
29
�
'0
(1
+2
,3
-4
.5
16
27
38
49
710
811
912
:13
=14
>15
?16
@17
A18
B19
20
21
 
�
Cnon_trainable_variables

Dlayers
Elayer_metrics
	variables
trainable_variables
Flayer_regularization_losses
Gmetrics
regularization_losses
 
R
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
�
Laxis
	'gamma
(beta
)moving_mean
*moving_variance
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
h

+kernel
,bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
�
Uaxis
	-gamma
.beta
/moving_mean
0moving_variance
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
h

1kernel
2bias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
R
^regularization_losses
_	variables
`trainable_variables
a	keras_api
�
baxis
	3gamma
4beta
5moving_mean
6moving_variance
cregularization_losses
d	variables
etrainable_variables
f	keras_api
h

7kernel
8bias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
�
kaxis
	9gamma
:beta
;moving_mean
<moving_variance
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
h

=kernel
>bias
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
R
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
h

?kernel
@bias
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
R
|regularization_losses
}	variables
~trainable_variables
	keras_api
l

Akernel
Bbias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
@25
A26
B27
�
'0
(1
+2
,3
-4
.5
16
27
38
49
710
811
912
:13
=14
>15
?16
@17
A18
B19
 
�
�non_trainable_variables
�layers
�layer_metrics
	variables
trainable_variables
 �layer_regularization_losses
�metrics
regularization_losses
NL
VARIABLE_VALUEdense_14/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_14/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
�non_trainable_variables
�layer_metrics
regularization_losses
	variables
 trainable_variables
 �layer_regularization_losses
�metrics
�layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_16/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_16/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_16/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_16/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_16/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_16/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_17/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_17/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_17/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_17/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_17/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_17/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_18/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_18/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_18/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_18/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_18/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_18/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_19/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_19/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_19/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_19/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_19/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_19/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_12/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_12/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_13/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_13/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
8
)0
*1
/2
03
54
65
;6
<7

0
1
 
 

�0
�1
 
 
 
�
�non_trainable_variables
�layer_metrics
Hregularization_losses
I	variables
Jtrainable_variables
 �layer_regularization_losses
�metrics
�layers
 
 

'0
(1
)2
*3

'0
(1
�
�non_trainable_variables
�layer_metrics
Mregularization_losses
N	variables
Otrainable_variables
 �layer_regularization_losses
�metrics
�layers
 

+0
,1

+0
,1
�
�non_trainable_variables
�layer_metrics
Qregularization_losses
R	variables
Strainable_variables
 �layer_regularization_losses
�metrics
�layers
 
 

-0
.1
/2
03

-0
.1
�
�non_trainable_variables
�layer_metrics
Vregularization_losses
W	variables
Xtrainable_variables
 �layer_regularization_losses
�metrics
�layers
 

10
21

10
21
�
�non_trainable_variables
�layer_metrics
Zregularization_losses
[	variables
\trainable_variables
 �layer_regularization_losses
�metrics
�layers
 
 
 
�
�non_trainable_variables
�layer_metrics
^regularization_losses
_	variables
`trainable_variables
 �layer_regularization_losses
�metrics
�layers
 
 

30
41
52
63

30
41
�
�non_trainable_variables
�layer_metrics
cregularization_losses
d	variables
etrainable_variables
 �layer_regularization_losses
�metrics
�layers
 

70
81

70
81
�
�non_trainable_variables
�layer_metrics
gregularization_losses
h	variables
itrainable_variables
 �layer_regularization_losses
�metrics
�layers
 
 

90
:1
;2
<3

90
:1
�
�non_trainable_variables
�layer_metrics
lregularization_losses
m	variables
ntrainable_variables
 �layer_regularization_losses
�metrics
�layers
 

=0
>1

=0
>1
�
�non_trainable_variables
�layer_metrics
pregularization_losses
q	variables
rtrainable_variables
 �layer_regularization_losses
�metrics
�layers
 
 
 
�
�non_trainable_variables
�layer_metrics
tregularization_losses
u	variables
vtrainable_variables
 �layer_regularization_losses
�metrics
�layers
 

?0
@1

?0
@1
�
�non_trainable_variables
�layer_metrics
xregularization_losses
y	variables
ztrainable_variables
 �layer_regularization_losses
�metrics
�layers
 
 
 
�
�non_trainable_variables
�layer_metrics
|regularization_losses
}	variables
~trainable_variables
 �layer_regularization_losses
�metrics
�layers
 

A0
B1

A0
B1
�
�non_trainable_variables
�layer_metrics
�regularization_losses
�	variables
�trainable_variables
 �layer_regularization_losses
�metrics
�layers
 
 
 
�
�non_trainable_variables
�layer_metrics
�regularization_losses
�	variables
�trainable_variables
 �layer_regularization_losses
�metrics
�layers
8
)0
*1
/2
03
54
65
;6
<7
n
	0

1
2
3
4
5
6
7
8
9
10
11
12
13
14
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
 
 
 
 
 

)0
*1
 
 
 
 
 
 
 
 
 

/0
01
 
 
 
 
 
 
 
 
 
 
 
 
 
 

50
61
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
qo
VARIABLE_VALUEAdam/dense_14/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_14/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_16/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_16/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_16/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_17/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_17/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_17/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_18/beta/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_18/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_18/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_19/beta/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_19/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_19/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_12/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_12/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_13/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_13/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_14/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_14/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_16/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_16/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_16/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_17/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_17/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_17/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_18/beta/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_18/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_18/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_19/beta/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_19/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_19/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_12/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_12/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_13/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_13/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������KK*
dtype0*$
shape:���������KK
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_16/kernelconv2d_16/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_19/kernelconv2d_19/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1764275
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
� 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_16/beta/m/Read/ReadVariableOp+Adam/conv2d_16/kernel/m/Read/ReadVariableOp)Adam/conv2d_16/bias/m/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_17/beta/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_18/beta/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_19/beta/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_16/beta/v/Read/ReadVariableOp+Adam/conv2d_16/kernel/v/Read/ReadVariableOp)Adam/conv2d_16/bias/v/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_17/beta/v/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_18/beta/v/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_19/beta/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpConst*`
TinY
W2U	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_1766981
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_16/kernelconv2d_16/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_19/kernelconv2d_19/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biastotalcounttotal_1count_1Adam/dense_14/kernel/mAdam/dense_14/bias/m#Adam/batch_normalization_16/gamma/m"Adam/batch_normalization_16/beta/mAdam/conv2d_16/kernel/mAdam/conv2d_16/bias/m#Adam/batch_normalization_17/gamma/m"Adam/batch_normalization_17/beta/mAdam/conv2d_17/kernel/mAdam/conv2d_17/bias/m#Adam/batch_normalization_18/gamma/m"Adam/batch_normalization_18/beta/mAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/m#Adam/batch_normalization_19/gamma/m"Adam/batch_normalization_19/beta/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/v#Adam/batch_normalization_16/gamma/v"Adam/batch_normalization_16/beta/vAdam/conv2d_16/kernel/vAdam/conv2d_16/bias/v#Adam/batch_normalization_17/gamma/v"Adam/batch_normalization_17/beta/vAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/v#Adam/batch_normalization_18/gamma/v"Adam/batch_normalization_18/beta/vAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/v#Adam/batch_normalization_19/gamma/v"Adam/batch_normalization_19/beta/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v*_
TinX
V2T*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1767240��%
�
�
"__inference__wrapped_model_1761933
input_1
cnn3_1761871:
cnn3_1761873:
cnn3_1761875:
cnn3_1761877:&
cnn3_1761879: 
cnn3_1761881: 
cnn3_1761883: 
cnn3_1761885: 
cnn3_1761887: 
cnn3_1761889: '
cnn3_1761891: �
cnn3_1761893:	�
cnn3_1761895:	�
cnn3_1761897:	�
cnn3_1761899:	�
cnn3_1761901:	�(
cnn3_1761903:��
cnn3_1761905:	�
cnn3_1761907:	�
cnn3_1761909:	�
cnn3_1761911:	�
cnn3_1761913:	�(
cnn3_1761915:��
cnn3_1761917:	�!
cnn3_1761919:��*�
cnn3_1761921:	� 
cnn3_1761923:
��
cnn3_1761925:	�
cnn3_1761927:	�
cnn3_1761929:
identity��CNN3/StatefulPartitionedCall�
CNN3/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn3_1761871cnn3_1761873cnn3_1761875cnn3_1761877cnn3_1761879cnn3_1761881cnn3_1761883cnn3_1761885cnn3_1761887cnn3_1761889cnn3_1761891cnn3_1761893cnn3_1761895cnn3_1761897cnn3_1761899cnn3_1761901cnn3_1761903cnn3_1761905cnn3_1761907cnn3_1761909cnn3_1761911cnn3_1761913cnn3_1761915cnn3_1761917cnn3_1761919cnn3_1761921cnn3_1761923cnn3_1761925cnn3_1761927cnn3_1761929**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8� *!
fR
__inference_call_15318822
CNN3/StatefulPartitionedCall�
IdentityIdentity%CNN3/StatefulPartitionedCall:output:0^CNN3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
CNN3/StatefulPartitionedCallCNN3/StatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
F
*__inference_lambda_4_layer_call_fn_1765954

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
:���������KK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_17624642
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������KK:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
E__inference_dense_13_layer_call_and_return_conditional_losses_1766640

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_1766654

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_4_layer_call_fn_1765730
lambda_4_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_17627362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_4_input
�
�
__inference_loss_fn_1_1766698O
:dense_12_kernel_regularizer_square_readvariableop_resource:��*�
identity��1dense_12/kernel/Regularizer/Square/ReadVariableOp�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_12_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
IdentityIdentity#dense_12/kernel/Regularizer/mul:z:02^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766277

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
*__inference_dense_14_layer_call_fn_1765933

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_17636392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_1762655

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������*2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�

�
E__inference_dense_14_layer_call_and_return_conditional_losses_1765924

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_13_layer_call_fn_1766649

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_17627042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766133

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766439

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�%
 __inference__traced_save_1766981
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_m_read_readvariableop6
2savev2_adam_conv2d_16_kernel_m_read_readvariableop4
0savev2_adam_conv2d_16_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_v_read_readvariableop6
2savev2_adam_conv2d_16_kernel_v_read_readvariableop4
0savev2_adam_conv2d_16_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_v_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_v_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�%
value�%B�$TB)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�
value�B�TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop>savev2_adam_batch_normalization_16_gamma_m_read_readvariableop=savev2_adam_batch_normalization_16_beta_m_read_readvariableop2savev2_adam_conv2d_16_kernel_m_read_readvariableop0savev2_adam_conv2d_16_bias_m_read_readvariableop>savev2_adam_batch_normalization_17_gamma_m_read_readvariableop=savev2_adam_batch_normalization_17_beta_m_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop>savev2_adam_batch_normalization_18_gamma_m_read_readvariableop=savev2_adam_batch_normalization_18_beta_m_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_19_gamma_m_read_readvariableop=savev2_adam_batch_normalization_19_beta_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop>savev2_adam_batch_normalization_16_gamma_v_read_readvariableop=savev2_adam_batch_normalization_16_beta_v_read_readvariableop2savev2_adam_conv2d_16_kernel_v_read_readvariableop0savev2_adam_conv2d_16_bias_v_read_readvariableop>savev2_adam_batch_normalization_17_gamma_v_read_readvariableop=savev2_adam_batch_normalization_17_beta_v_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableop>savev2_adam_batch_normalization_18_gamma_v_read_readvariableop=savev2_adam_batch_normalization_18_beta_v_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_19_gamma_v_read_readvariableop=savev2_adam_batch_normalization_19_beta_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *b
dtypesX
V2T	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:: : : : : ::::: : : : : : : �:�:�:�:�:�:��:�:�:�:�:�:��:�:��*�:�:
��:�: : : : :	�:::: : : : : �:�:�:�:��:�:�:�:��:�:��*�:�:
��:�:	�:::: : : : : �:�:�:�:��:�:�:�:��:�:��*�:�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: �:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:' #
!
_output_shapes
:��*�:!!

_output_shapes	
:�:&""
 
_output_shapes
:
��:!#

_output_shapes	
:�:$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :%(!

_output_shapes
:	�: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: : .

_output_shapes
: : /

_output_shapes
: :-0)
'
_output_shapes
: �:!1

_output_shapes	
:�:!2

_output_shapes	
:�:!3

_output_shapes	
:�:.4*
(
_output_shapes
:��:!5

_output_shapes	
:�:!6

_output_shapes	
:�:!7

_output_shapes	
:�:.8*
(
_output_shapes
:��:!9

_output_shapes	
:�:':#
!
_output_shapes
:��*�:!;

_output_shapes	
:�:&<"
 
_output_shapes
:
��:!=

_output_shapes	
:�:%>!

_output_shapes
:	�: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
: : C

_output_shapes
: : D

_output_shapes
: : E

_output_shapes
: :-F)
'
_output_shapes
: �:!G

_output_shapes	
:�:!H

_output_shapes	
:�:!I

_output_shapes	
:�:.J*
(
_output_shapes
:��:!K

_output_shapes	
:�:!L

_output_shapes	
:�:!M

_output_shapes	
:�:.N*
(
_output_shapes
:��:!O

_output_shapes	
:�:'P#
!
_output_shapes
:��*�:!Q

_output_shapes	
:�:&R"
 
_output_shapes
:
��:!S

_output_shapes	
:�:T

_output_shapes
: 
��
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765526
lambda_4_input<
.batch_normalization_16_readvariableop_resource:>
0batch_normalization_16_readvariableop_1_resource:M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_16_conv2d_readvariableop_resource: 7
)conv2d_16_biasadd_readvariableop_resource: <
.batch_normalization_17_readvariableop_resource: >
0batch_normalization_17_readvariableop_1_resource: M
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: C
(conv2d_17_conv2d_readvariableop_resource: �8
)conv2d_17_biasadd_readvariableop_resource:	�=
.batch_normalization_18_readvariableop_resource:	�?
0batch_normalization_18_readvariableop_1_resource:	�N
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_18_conv2d_readvariableop_resource:��8
)conv2d_18_biasadd_readvariableop_resource:	�=
.batch_normalization_19_readvariableop_resource:	�?
0batch_normalization_19_readvariableop_1_resource:	�N
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_19_conv2d_readvariableop_resource:��8
)conv2d_19_biasadd_readvariableop_resource:	�<
'dense_12_matmul_readvariableop_resource:��*�7
(dense_12_biasadd_readvariableop_resource:	�;
'dense_13_matmul_readvariableop_resource:
��7
(dense_13_biasadd_readvariableop_resource:	�
identity��6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_16/ReadVariableOp�'batch_normalization_16/ReadVariableOp_1�6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_17/ReadVariableOp�'batch_normalization_17/ReadVariableOp_1�6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_18/ReadVariableOp�'batch_normalization_18/ReadVariableOp_1�6batch_normalization_19/FusedBatchNormV3/ReadVariableOp�8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_19/ReadVariableOp�'batch_normalization_19/ReadVariableOp_1� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�2conv2d_16/kernel/Regularizer/Square/ReadVariableOp� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stack�
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1�
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2�
lambda_4/strided_sliceStridedSlicelambda_4_input%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_4/strided_slice�
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp�
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1�
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_16/Conv2D/ReadVariableOp�
conv2d_16/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_16/Conv2D�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_16/Relu�
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_17/ReadVariableOp�
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_17/ReadVariableOp_1�
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_17/Conv2D/ReadVariableOp�
conv2d_17/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
conv2d_17/Conv2D�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
conv2d_17/Relu�
max_pooling2d_4/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool�
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_18/ReadVariableOp�
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_18/ReadVariableOp_1�
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_18/FusedBatchNormV3�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_18/Conv2D�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_18/BiasAdd
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_18/Relu�
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_19/ReadVariableOp�
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_19/ReadVariableOp_1�
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_18/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_19/FusedBatchNormV3�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2DConv2D+batch_normalization_19/FusedBatchNormV3:y:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_19/Conv2D�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
flatten_4/Const�
flatten_4/ReshapeReshapeconv2d_19/Relu:activations:0flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2
flatten_4/Reshape�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_12/Relu�
dropout_8/IdentityIdentitydense_12/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_8/Identity�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMuldropout_8/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_13/Relu�
dropout_9/IdentityIdentitydense_13/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_9/Identity�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentitydropout_9/Identity:output:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp3^conv2d_16/kernel/Regularizer/Square/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_4_input
�
�
+__inference_conv2d_16_layer_call_fn_1766115

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_16_layer_call_and_return_conditional_losses_17625102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_18_layer_call_fn_1766357

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17622632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_1766687U
;conv2d_16_kernel_regularizer_square_readvariableop_resource: 
identity��2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_16_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
IdentityIdentity$conv2d_16/kernel/Regularizer/mul:z:03^conv2d_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp
�
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_1762685

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_19_layer_call_and_return_conditional_losses_1766538

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_19_layer_call_fn_1766488

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17623452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1761999

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_17_layer_call_fn_1766226

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17625332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1762191

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766295

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
&__inference_CNN3_layer_call_fn_1764977

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_17636642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1766250

inputs9
conv2d_readvariableop_resource: �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������KK�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1762578

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1765977

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
�
__inference_call_1535237

inputsI
;sequential_4_batch_normalization_16_readvariableop_resource:K
=sequential_4_batch_normalization_16_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_16_conv2d_readvariableop_resource: D
6sequential_4_conv2d_16_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_17_readvariableop_resource: K
=sequential_4_batch_normalization_17_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_4_conv2d_17_conv2d_readvariableop_resource: �E
6sequential_4_conv2d_17_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_18_readvariableop_resource:	�L
=sequential_4_batch_normalization_18_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_18_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_18_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_19_readvariableop_resource:	�L
=sequential_4_batch_normalization_19_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_19_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_19_biasadd_readvariableop_resource:	�I
4sequential_4_dense_12_matmul_readvariableop_resource:��*�D
5sequential_4_dense_12_biasadd_readvariableop_resource:	�H
4sequential_4_dense_13_matmul_readvariableop_resource:
��D
5sequential_4_dense_13_biasadd_readvariableop_resource:	�:
'dense_14_matmul_readvariableop_resource:	�6
(dense_14_biasadd_readvariableop_resource:
identity��dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_16/ReadVariableOp�4sequential_4/batch_normalization_16/ReadVariableOp_1�Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_17/ReadVariableOp�4sequential_4/batch_normalization_17/ReadVariableOp_1�Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_18/ReadVariableOp�4sequential_4/batch_normalization_18/ReadVariableOp_1�Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_19/ReadVariableOp�4sequential_4/batch_normalization_19/ReadVariableOp_1�-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�,sequential_4/conv2d_16/Conv2D/ReadVariableOp�-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�,sequential_4/conv2d_17/Conv2D/ReadVariableOp�-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�,sequential_4/conv2d_18/Conv2D/ReadVariableOp�-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�,sequential_4/conv2d_19/Conv2D/ReadVariableOp�,sequential_4/dense_12/BiasAdd/ReadVariableOp�+sequential_4/dense_12/MatMul/ReadVariableOp�,sequential_4/dense_13/BiasAdd/ReadVariableOp�+sequential_4/dense_13/MatMul/ReadVariableOp�
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack�
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1�
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2�
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_slice�
2sequential_4/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_4/batch_normalization_16/ReadVariableOp�
4sequential_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_4/batch_normalization_16/ReadVariableOp_1�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:0:sequential_4/batch_normalization_16/ReadVariableOp:value:0<sequential_4/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_16/FusedBatchNormV3�
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_16/Conv2D/ReadVariableOp�
sequential_4/conv2d_16/Conv2DConv2D8sequential_4/batch_normalization_16/FusedBatchNormV3:y:04sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_4/conv2d_16/Conv2D�
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_4/conv2d_16/BiasAdd�
sequential_4/conv2d_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_4/conv2d_16/Relu�
2sequential_4/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_4/batch_normalization_17/ReadVariableOp�
4sequential_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_4/batch_normalization_17/ReadVariableOp_1�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_16/Relu:activations:0:sequential_4/batch_normalization_17/ReadVariableOp:value:0<sequential_4/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_17/FusedBatchNormV3�
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_4/conv2d_17/Conv2D/ReadVariableOp�
sequential_4/conv2d_17/Conv2DConv2D8sequential_4/batch_normalization_17/FusedBatchNormV3:y:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�KK�*
paddingSAME*
strides
2
sequential_4/conv2d_17/Conv2D�
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�KK�2 
sequential_4/conv2d_17/BiasAdd�
sequential_4/conv2d_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*(
_output_shapes
:�KK�2
sequential_4/conv2d_17/Relu�
$sequential_4/max_pooling2d_4/MaxPoolMaxPool)sequential_4/conv2d_17/Relu:activations:0*(
_output_shapes
:�%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_4/MaxPool�
2sequential_4/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_18/ReadVariableOp�
4sequential_4/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_18/ReadVariableOp_1�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_4/max_pooling2d_4/MaxPool:output:0:sequential_4/batch_normalization_18/ReadVariableOp:value:0<sequential_4/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*H
_output_shapes6
4:�%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_18/FusedBatchNormV3�
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_18/Conv2D/ReadVariableOp�
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_18/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_4/conv2d_18/Conv2D�
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_4/conv2d_18/BiasAdd�
sequential_4/conv2d_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_4/conv2d_18/Relu�
2sequential_4/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_19/ReadVariableOp�
4sequential_4/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_19/ReadVariableOp_1�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_18/Relu:activations:0:sequential_4/batch_normalization_19/ReadVariableOp:value:0<sequential_4/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*H
_output_shapes6
4:�%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_19/FusedBatchNormV3�
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_19/Conv2D/ReadVariableOp�
sequential_4/conv2d_19/Conv2DConv2D8sequential_4/batch_normalization_19/FusedBatchNormV3:y:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_4/conv2d_19/Conv2D�
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_4/conv2d_19/BiasAdd�
sequential_4/conv2d_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_4/conv2d_19/Relu�
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_4/flatten_4/Const�
sequential_4/flatten_4/ReshapeReshape)sequential_4/conv2d_19/Relu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*!
_output_shapes
:���*2 
sequential_4/flatten_4/Reshape�
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp�
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_12/MatMul�
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp�
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_12/BiasAdd�
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_12/Relu�
sequential_4/dropout_8/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0* 
_output_shapes
:
��2!
sequential_4/dropout_8/Identity�
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp�
sequential_4/dense_13/MatMulMatMul(sequential_4/dropout_8/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_13/MatMul�
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp�
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_13/BiasAdd�
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_13/Relu�
sequential_4/dropout_9/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0* 
_output_shapes
:
��2!
sequential_4/dropout_9/Identity�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMul(sequential_4/dropout_9/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_14/BiasAddt
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_14/Softmax�
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpD^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_16/ReadVariableOp5^sequential_4/batch_normalization_16/ReadVariableOp_1D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_17/ReadVariableOp5^sequential_4/batch_normalization_17/ReadVariableOp_1D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_18/ReadVariableOp5^sequential_4/batch_normalization_18/ReadVariableOp_1D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_19/ReadVariableOp5^sequential_4/batch_normalization_19/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:�KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_16/ReadVariableOp2sequential_4/batch_normalization_16/ReadVariableOp2l
4sequential_4/batch_normalization_16/ReadVariableOp_14sequential_4/batch_normalization_16/ReadVariableOp_12�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_17/ReadVariableOp2sequential_4/batch_normalization_17/ReadVariableOp2l
4sequential_4/batch_normalization_17/ReadVariableOp_14sequential_4/batch_normalization_17/ReadVariableOp_12�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_18/ReadVariableOp2sequential_4/batch_normalization_18/ReadVariableOp2l
4sequential_4/batch_normalization_18/ReadVariableOp_14sequential_4/batch_normalization_18/ReadVariableOp_12�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_19/ReadVariableOp2sequential_4/batch_normalization_19/ReadVariableOp2l
4sequential_4/batch_normalization_19/ReadVariableOp_14sequential_4/batch_normalization_19/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_16_layer_call_fn_1766070

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17624832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
&__inference_CNN3_layer_call_fn_1765107
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_17638882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766187

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
e
F__inference_dropout_9_layer_call_and_return_conditional_losses_1766666

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
F__inference_dropout_8_layer_call_and_return_conditional_losses_1766607

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_1765949

inputs
identity�
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������KK:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_1762464

inputs
identity�
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������KK:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766475

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1766031

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1762345

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_1765941

inputs
identity�
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������KK:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1762125

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_1762715

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_CNN3_layer_call_fn_1765042

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_17638882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
G
+__inference_dropout_9_layer_call_fn_1766671

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_17627152
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_18_layer_call_fn_1766383

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17629642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
G
+__inference_dropout_8_layer_call_fn_1766612

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_17626852
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_19_layer_call_fn_1766547

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_19_layer_call_and_return_conditional_losses_17626432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_1766709N
:dense_13_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_13/kernel/Regularizer/Square/ReadVariableOp�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_13_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentity#dense_13/kernel/Regularizer/mul:z:02^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp
�r
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1763258

inputs,
batch_normalization_16_1763169:,
batch_normalization_16_1763171:,
batch_normalization_16_1763173:,
batch_normalization_16_1763175:+
conv2d_16_1763178: 
conv2d_16_1763180: ,
batch_normalization_17_1763183: ,
batch_normalization_17_1763185: ,
batch_normalization_17_1763187: ,
batch_normalization_17_1763189: ,
conv2d_17_1763192: � 
conv2d_17_1763194:	�-
batch_normalization_18_1763198:	�-
batch_normalization_18_1763200:	�-
batch_normalization_18_1763202:	�-
batch_normalization_18_1763204:	�-
conv2d_18_1763207:�� 
conv2d_18_1763209:	�-
batch_normalization_19_1763212:	�-
batch_normalization_19_1763214:	�-
batch_normalization_19_1763216:	�-
batch_normalization_19_1763218:	�-
conv2d_19_1763221:�� 
conv2d_19_1763223:	�%
dense_12_1763227:��*�
dense_12_1763229:	�$
dense_13_1763233:
��
dense_13_1763235:	�
identity��.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall�2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall� dense_12/StatefulPartitionedCall�1dense_12/kernel/Regularizer/Square/ReadVariableOp� dense_13/StatefulPartitionedCall�1dense_13/kernel/Regularizer/Square/ReadVariableOp�!dropout_8/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_17630992
lambda_4/PartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0batch_normalization_16_1763169batch_normalization_16_1763171batch_normalization_16_1763173batch_normalization_16_1763175*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_176307220
.batch_normalization_16/StatefulPartitionedCall�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_16_1763178conv2d_16_1763180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_16_layer_call_and_return_conditional_losses_17625102#
!conv2d_16/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_17_1763183batch_normalization_17_1763185batch_normalization_17_1763187batch_normalization_17_1763189*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_176301820
.batch_normalization_17/StatefulPartitionedCall�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0conv2d_17_1763192conv2d_17_1763194*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������KK�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_17625542#
!conv2d_17/StatefulPartitionedCall�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17621912!
max_pooling2d_4/PartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_18_1763198batch_normalization_18_1763200batch_normalization_18_1763202batch_normalization_18_1763204*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_176296420
.batch_normalization_18/StatefulPartitionedCall�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_18_1763207conv2d_18_1763209*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_18_layer_call_and_return_conditional_losses_17625992#
!conv2d_18/StatefulPartitionedCall�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_19_1763212batch_normalization_19_1763214batch_normalization_19_1763216batch_normalization_19_1763218*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_176291020
.batch_normalization_19/StatefulPartitionedCall�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0conv2d_19_1763221conv2d_19_1763223*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_19_layer_call_and_return_conditional_losses_17626432#
!conv2d_19/StatefulPartitionedCall�
flatten_4/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_17626552
flatten_4/PartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_1763227dense_12_1763229*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_17626742"
 dense_12/StatefulPartitionedCall�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_17628482#
!dropout_8/StatefulPartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_13_1763233dense_13_1763235*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_17627042"
 dense_13/StatefulPartitionedCall�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_17628152#
!dropout_9/StatefulPartitionedCall�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_1763178*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_1763227*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_1763233* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentity*dropout_9/StatefulPartitionedCall:output:0/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall3^conv2d_16/kernel/Regularizer/Square/ReadVariableOp"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_19_layer_call_fn_1766514

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17626222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
��
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765669
lambda_4_input<
.batch_normalization_16_readvariableop_resource:>
0batch_normalization_16_readvariableop_1_resource:M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_16_conv2d_readvariableop_resource: 7
)conv2d_16_biasadd_readvariableop_resource: <
.batch_normalization_17_readvariableop_resource: >
0batch_normalization_17_readvariableop_1_resource: M
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: C
(conv2d_17_conv2d_readvariableop_resource: �8
)conv2d_17_biasadd_readvariableop_resource:	�=
.batch_normalization_18_readvariableop_resource:	�?
0batch_normalization_18_readvariableop_1_resource:	�N
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_18_conv2d_readvariableop_resource:��8
)conv2d_18_biasadd_readvariableop_resource:	�=
.batch_normalization_19_readvariableop_resource:	�?
0batch_normalization_19_readvariableop_1_resource:	�N
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_19_conv2d_readvariableop_resource:��8
)conv2d_19_biasadd_readvariableop_resource:	�<
'dense_12_matmul_readvariableop_resource:��*�7
(dense_12_biasadd_readvariableop_resource:	�;
'dense_13_matmul_readvariableop_resource:
��7
(dense_13_biasadd_readvariableop_resource:	�
identity��%batch_normalization_16/AssignNewValue�'batch_normalization_16/AssignNewValue_1�6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_16/ReadVariableOp�'batch_normalization_16/ReadVariableOp_1�%batch_normalization_17/AssignNewValue�'batch_normalization_17/AssignNewValue_1�6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_17/ReadVariableOp�'batch_normalization_17/ReadVariableOp_1�%batch_normalization_18/AssignNewValue�'batch_normalization_18/AssignNewValue_1�6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_18/ReadVariableOp�'batch_normalization_18/ReadVariableOp_1�%batch_normalization_19/AssignNewValue�'batch_normalization_19/AssignNewValue_1�6batch_normalization_19/FusedBatchNormV3/ReadVariableOp�8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_19/ReadVariableOp�'batch_normalization_19/ReadVariableOp_1� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�2conv2d_16/kernel/Regularizer/Square/ReadVariableOp� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stack�
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1�
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2�
lambda_4/strided_sliceStridedSlicelambda_4_input%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_4/strided_slice�
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp�
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1�
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_16/FusedBatchNormV3�
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValue�
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_16/Conv2D/ReadVariableOp�
conv2d_16/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_16/Conv2D�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_16/Relu�
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_17/ReadVariableOp�
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_17/ReadVariableOp_1�
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_17/FusedBatchNormV3�
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue�
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_17/Conv2D/ReadVariableOp�
conv2d_17/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
conv2d_17/Conv2D�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
conv2d_17/Relu�
max_pooling2d_4/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool�
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_18/ReadVariableOp�
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_18/ReadVariableOp_1�
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_18/FusedBatchNormV3�
%batch_normalization_18/AssignNewValueAssignVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource4batch_normalization_18/FusedBatchNormV3:batch_mean:07^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_18/AssignNewValue�
'batch_normalization_18/AssignNewValue_1AssignVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_18/FusedBatchNormV3:batch_variance:09^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_18/AssignNewValue_1�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_18/Conv2D�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_18/BiasAdd
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_18/Relu�
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_19/ReadVariableOp�
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_19/ReadVariableOp_1�
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_18/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_19/FusedBatchNormV3�
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_19/AssignNewValue�
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_19/AssignNewValue_1�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2DConv2D+batch_normalization_19/FusedBatchNormV3:y:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_19/Conv2D�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
flatten_4/Const�
flatten_4/ReshapeReshapeconv2d_19/Relu:activations:0flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2
flatten_4/Reshape�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_12/Reluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_8/dropout/Const�
dropout_8/dropout/MulMuldense_12/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_8/dropout/Mul}
dropout_8/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape�
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform�
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_8/dropout/GreaterEqual/y�
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2 
dropout_8/dropout/GreaterEqual�
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_8/dropout/Cast�
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_8/dropout/Mul_1�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMuldropout_8/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_13/Reluw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_9/dropout/Const�
dropout_9/dropout/MulMuldense_13/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_9/dropout/Mul}
dropout_9/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform�
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_9/dropout/GreaterEqual/y�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2 
dropout_9/dropout/GreaterEqual�
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_9/dropout/Cast�
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_9/dropout/Mul_1�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentitydropout_9/dropout/Mul_1:z:0&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1&^batch_normalization_18/AssignNewValue(^batch_normalization_18/AssignNewValue_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp3^conv2d_16/kernel/Regularizer/Square/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12N
%batch_normalization_18/AssignNewValue%batch_normalization_18/AssignNewValue2R
'batch_normalization_18/AssignNewValue_1'batch_normalization_18/AssignNewValue_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_4_input
�
�
E__inference_dense_13_layer_call_and_return_conditional_losses_1762704

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_16_layer_call_fn_1766044

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17619552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_18_layer_call_fn_1766370

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17625782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766151

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_17_layer_call_fn_1766213

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17621252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1762081

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1762263

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_18_layer_call_and_return_conditional_losses_1762599

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_18_layer_call_fn_1766344

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17622192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_17_layer_call_fn_1766259

inputs"
unknown: �
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������KK�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_17625542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������KK�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_4_layer_call_fn_1762197

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17621912
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�9
�
A__inference_CNN3_layer_call_and_return_conditional_losses_1763888

inputs"
sequential_4_1763807:"
sequential_4_1763809:"
sequential_4_1763811:"
sequential_4_1763813:.
sequential_4_1763815: "
sequential_4_1763817: "
sequential_4_1763819: "
sequential_4_1763821: "
sequential_4_1763823: "
sequential_4_1763825: /
sequential_4_1763827: �#
sequential_4_1763829:	�#
sequential_4_1763831:	�#
sequential_4_1763833:	�#
sequential_4_1763835:	�#
sequential_4_1763837:	�0
sequential_4_1763839:��#
sequential_4_1763841:	�#
sequential_4_1763843:	�#
sequential_4_1763845:	�#
sequential_4_1763847:	�#
sequential_4_1763849:	�0
sequential_4_1763851:��#
sequential_4_1763853:	�)
sequential_4_1763855:��*�#
sequential_4_1763857:	�(
sequential_4_1763859:
��#
sequential_4_1763861:	�#
dense_14_1763864:	�
dense_14_1763866:
identity��2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp� dense_14/StatefulPartitionedCall�$sequential_4/StatefulPartitionedCall�
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_1763807sequential_4_1763809sequential_4_1763811sequential_4_1763813sequential_4_1763815sequential_4_1763817sequential_4_1763819sequential_4_1763821sequential_4_1763823sequential_4_1763825sequential_4_1763827sequential_4_1763829sequential_4_1763831sequential_4_1763833sequential_4_1763835sequential_4_1763837sequential_4_1763839sequential_4_1763841sequential_4_1763843sequential_4_1763845sequential_4_1763847sequential_4_1763849sequential_4_1763851sequential_4_1763853sequential_4_1763855sequential_4_1763857sequential_4_1763859sequential_4_1763861*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*6
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_17632582&
$sequential_4/StatefulPartitionedCall�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0dense_14_1763864dense_14_1763866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_17636392"
 dense_14/StatefulPartitionedCall�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_1763815*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_1763855*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_1763859* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentity)dense_14/StatefulPartitionedCall:output:03^conv2d_16/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
��
�
__inference_call_1531882

inputsI
;sequential_4_batch_normalization_16_readvariableop_resource:K
=sequential_4_batch_normalization_16_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_16_conv2d_readvariableop_resource: D
6sequential_4_conv2d_16_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_17_readvariableop_resource: K
=sequential_4_batch_normalization_17_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_4_conv2d_17_conv2d_readvariableop_resource: �E
6sequential_4_conv2d_17_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_18_readvariableop_resource:	�L
=sequential_4_batch_normalization_18_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_18_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_18_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_19_readvariableop_resource:	�L
=sequential_4_batch_normalization_19_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_19_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_19_biasadd_readvariableop_resource:	�I
4sequential_4_dense_12_matmul_readvariableop_resource:��*�D
5sequential_4_dense_12_biasadd_readvariableop_resource:	�H
4sequential_4_dense_13_matmul_readvariableop_resource:
��D
5sequential_4_dense_13_biasadd_readvariableop_resource:	�:
'dense_14_matmul_readvariableop_resource:	�6
(dense_14_biasadd_readvariableop_resource:
identity��dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_16/ReadVariableOp�4sequential_4/batch_normalization_16/ReadVariableOp_1�Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_17/ReadVariableOp�4sequential_4/batch_normalization_17/ReadVariableOp_1�Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_18/ReadVariableOp�4sequential_4/batch_normalization_18/ReadVariableOp_1�Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_19/ReadVariableOp�4sequential_4/batch_normalization_19/ReadVariableOp_1�-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�,sequential_4/conv2d_16/Conv2D/ReadVariableOp�-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�,sequential_4/conv2d_17/Conv2D/ReadVariableOp�-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�,sequential_4/conv2d_18/Conv2D/ReadVariableOp�-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�,sequential_4/conv2d_19/Conv2D/ReadVariableOp�,sequential_4/dense_12/BiasAdd/ReadVariableOp�+sequential_4/dense_12/MatMul/ReadVariableOp�,sequential_4/dense_13/BiasAdd/ReadVariableOp�+sequential_4/dense_13/MatMul/ReadVariableOp�
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack�
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1�
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2�
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_slice�
2sequential_4/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_4/batch_normalization_16/ReadVariableOp�
4sequential_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_4/batch_normalization_16/ReadVariableOp_1�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:0:sequential_4/batch_normalization_16/ReadVariableOp:value:0<sequential_4/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_16/FusedBatchNormV3�
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_16/Conv2D/ReadVariableOp�
sequential_4/conv2d_16/Conv2DConv2D8sequential_4/batch_normalization_16/FusedBatchNormV3:y:04sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_4/conv2d_16/Conv2D�
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_4/conv2d_16/BiasAdd�
sequential_4/conv2d_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_4/conv2d_16/Relu�
2sequential_4/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_4/batch_normalization_17/ReadVariableOp�
4sequential_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_4/batch_normalization_17/ReadVariableOp_1�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_16/Relu:activations:0:sequential_4/batch_normalization_17/ReadVariableOp:value:0<sequential_4/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_17/FusedBatchNormV3�
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_4/conv2d_17/Conv2D/ReadVariableOp�
sequential_4/conv2d_17/Conv2DConv2D8sequential_4/batch_normalization_17/FusedBatchNormV3:y:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_4/conv2d_17/Conv2D�
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_4/conv2d_17/BiasAdd�
sequential_4/conv2d_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_4/conv2d_17/Relu�
$sequential_4/max_pooling2d_4/MaxPoolMaxPool)sequential_4/conv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_4/MaxPool�
2sequential_4/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_18/ReadVariableOp�
4sequential_4/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_18/ReadVariableOp_1�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_4/max_pooling2d_4/MaxPool:output:0:sequential_4/batch_normalization_18/ReadVariableOp:value:0<sequential_4/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_18/FusedBatchNormV3�
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_18/Conv2D/ReadVariableOp�
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_18/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_18/Conv2D�
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_18/BiasAdd�
sequential_4/conv2d_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_18/Relu�
2sequential_4/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_19/ReadVariableOp�
4sequential_4/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_19/ReadVariableOp_1�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_18/Relu:activations:0:sequential_4/batch_normalization_19/ReadVariableOp:value:0<sequential_4/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_19/FusedBatchNormV3�
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_19/Conv2D/ReadVariableOp�
sequential_4/conv2d_19/Conv2DConv2D8sequential_4/batch_normalization_19/FusedBatchNormV3:y:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_19/Conv2D�
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_19/BiasAdd�
sequential_4/conv2d_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_19/Relu�
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_4/flatten_4/Const�
sequential_4/flatten_4/ReshapeReshape)sequential_4/conv2d_19/Relu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_4/flatten_4/Reshape�
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp�
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/MatMul�
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp�
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/BiasAdd�
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/Relu�
sequential_4/dropout_8/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:����������2!
sequential_4/dropout_8/Identity�
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp�
sequential_4/dense_13/MatMulMatMul(sequential_4/dropout_8/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/MatMul�
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp�
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/BiasAdd�
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/Relu�
sequential_4/dropout_9/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:����������2!
sequential_4/dropout_9/Identity�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMul(sequential_4/dropout_9/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_14/Softmax�
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpD^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_16/ReadVariableOp5^sequential_4/batch_normalization_16/ReadVariableOp_1D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_17/ReadVariableOp5^sequential_4/batch_normalization_17/ReadVariableOp_1D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_18/ReadVariableOp5^sequential_4/batch_normalization_18/ReadVariableOp_1D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_19/ReadVariableOp5^sequential_4/batch_normalization_19/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_16/ReadVariableOp2sequential_4/batch_normalization_16/ReadVariableOp2l
4sequential_4/batch_normalization_16/ReadVariableOp_14sequential_4/batch_normalization_16/ReadVariableOp_12�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_17/ReadVariableOp2sequential_4/batch_normalization_17/ReadVariableOp2l
4sequential_4/batch_normalization_17/ReadVariableOp_14sequential_4/batch_normalization_17/ReadVariableOp_12�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_18/ReadVariableOp2sequential_4/batch_normalization_18/ReadVariableOp2l
4sequential_4/batch_normalization_18/ReadVariableOp_14sequential_4/batch_normalization_18/ReadVariableOp_12�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_19/ReadVariableOp2sequential_4/batch_normalization_19/ReadVariableOp2l
4sequential_4/batch_normalization_19/ReadVariableOp_14sequential_4/batch_normalization_19/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
��
�
__inference_call_1535355

inputsI
;sequential_4_batch_normalization_16_readvariableop_resource:K
=sequential_4_batch_normalization_16_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_16_conv2d_readvariableop_resource: D
6sequential_4_conv2d_16_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_17_readvariableop_resource: K
=sequential_4_batch_normalization_17_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_4_conv2d_17_conv2d_readvariableop_resource: �E
6sequential_4_conv2d_17_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_18_readvariableop_resource:	�L
=sequential_4_batch_normalization_18_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_18_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_18_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_19_readvariableop_resource:	�L
=sequential_4_batch_normalization_19_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_19_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_19_biasadd_readvariableop_resource:	�I
4sequential_4_dense_12_matmul_readvariableop_resource:��*�D
5sequential_4_dense_12_biasadd_readvariableop_resource:	�H
4sequential_4_dense_13_matmul_readvariableop_resource:
��D
5sequential_4_dense_13_biasadd_readvariableop_resource:	�:
'dense_14_matmul_readvariableop_resource:	�6
(dense_14_biasadd_readvariableop_resource:
identity��dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_16/ReadVariableOp�4sequential_4/batch_normalization_16/ReadVariableOp_1�Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_17/ReadVariableOp�4sequential_4/batch_normalization_17/ReadVariableOp_1�Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_18/ReadVariableOp�4sequential_4/batch_normalization_18/ReadVariableOp_1�Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_19/ReadVariableOp�4sequential_4/batch_normalization_19/ReadVariableOp_1�-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�,sequential_4/conv2d_16/Conv2D/ReadVariableOp�-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�,sequential_4/conv2d_17/Conv2D/ReadVariableOp�-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�,sequential_4/conv2d_18/Conv2D/ReadVariableOp�-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�,sequential_4/conv2d_19/Conv2D/ReadVariableOp�,sequential_4/dense_12/BiasAdd/ReadVariableOp�+sequential_4/dense_12/MatMul/ReadVariableOp�,sequential_4/dense_13/BiasAdd/ReadVariableOp�+sequential_4/dense_13/MatMul/ReadVariableOp�
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack�
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1�
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2�
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_slice�
2sequential_4/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_4/batch_normalization_16/ReadVariableOp�
4sequential_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_4/batch_normalization_16/ReadVariableOp_1�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:0:sequential_4/batch_normalization_16/ReadVariableOp:value:0<sequential_4/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_16/FusedBatchNormV3�
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_16/Conv2D/ReadVariableOp�
sequential_4/conv2d_16/Conv2DConv2D8sequential_4/batch_normalization_16/FusedBatchNormV3:y:04sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_4/conv2d_16/Conv2D�
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_4/conv2d_16/BiasAdd�
sequential_4/conv2d_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_4/conv2d_16/Relu�
2sequential_4/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_4/batch_normalization_17/ReadVariableOp�
4sequential_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_4/batch_normalization_17/ReadVariableOp_1�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_16/Relu:activations:0:sequential_4/batch_normalization_17/ReadVariableOp:value:0<sequential_4/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_17/FusedBatchNormV3�
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_4/conv2d_17/Conv2D/ReadVariableOp�
sequential_4/conv2d_17/Conv2DConv2D8sequential_4/batch_normalization_17/FusedBatchNormV3:y:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�KK�*
paddingSAME*
strides
2
sequential_4/conv2d_17/Conv2D�
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�KK�2 
sequential_4/conv2d_17/BiasAdd�
sequential_4/conv2d_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*(
_output_shapes
:�KK�2
sequential_4/conv2d_17/Relu�
$sequential_4/max_pooling2d_4/MaxPoolMaxPool)sequential_4/conv2d_17/Relu:activations:0*(
_output_shapes
:�%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_4/MaxPool�
2sequential_4/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_18/ReadVariableOp�
4sequential_4/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_18/ReadVariableOp_1�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_4/max_pooling2d_4/MaxPool:output:0:sequential_4/batch_normalization_18/ReadVariableOp:value:0<sequential_4/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*H
_output_shapes6
4:�%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_18/FusedBatchNormV3�
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_18/Conv2D/ReadVariableOp�
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_18/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_4/conv2d_18/Conv2D�
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_4/conv2d_18/BiasAdd�
sequential_4/conv2d_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_4/conv2d_18/Relu�
2sequential_4/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_19/ReadVariableOp�
4sequential_4/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_19/ReadVariableOp_1�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_18/Relu:activations:0:sequential_4/batch_normalization_19/ReadVariableOp:value:0<sequential_4/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*H
_output_shapes6
4:�%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_19/FusedBatchNormV3�
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_19/Conv2D/ReadVariableOp�
sequential_4/conv2d_19/Conv2DConv2D8sequential_4/batch_normalization_19/FusedBatchNormV3:y:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_4/conv2d_19/Conv2D�
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_4/conv2d_19/BiasAdd�
sequential_4/conv2d_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_4/conv2d_19/Relu�
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_4/flatten_4/Const�
sequential_4/flatten_4/ReshapeReshape)sequential_4/conv2d_19/Relu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*!
_output_shapes
:���*2 
sequential_4/flatten_4/Reshape�
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp�
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_12/MatMul�
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp�
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_12/BiasAdd�
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_12/Relu�
sequential_4/dropout_8/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0* 
_output_shapes
:
��2!
sequential_4/dropout_8/Identity�
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp�
sequential_4/dense_13/MatMulMatMul(sequential_4/dropout_8/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_13/MatMul�
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp�
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_13/BiasAdd�
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_4/dense_13/Relu�
sequential_4/dropout_9/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0* 
_output_shapes
:
��2!
sequential_4/dropout_9/Identity�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMul(sequential_4/dropout_9/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_14/BiasAddt
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_14/Softmax�
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpD^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_16/ReadVariableOp5^sequential_4/batch_normalization_16/ReadVariableOp_1D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_17/ReadVariableOp5^sequential_4/batch_normalization_17/ReadVariableOp_1D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_18/ReadVariableOp5^sequential_4/batch_normalization_18/ReadVariableOp_1D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_19/ReadVariableOp5^sequential_4/batch_normalization_19/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:�KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_16/ReadVariableOp2sequential_4/batch_normalization_16/ReadVariableOp2l
4sequential_4/batch_normalization_16/ReadVariableOp_14sequential_4/batch_normalization_16/ReadVariableOp_12�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_17/ReadVariableOp2sequential_4/batch_normalization_17/ReadVariableOp2l
4sequential_4/batch_normalization_17/ReadVariableOp_14sequential_4/batch_normalization_17/ReadVariableOp_12�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_18/ReadVariableOp2sequential_4/batch_normalization_18/ReadVariableOp2l
4sequential_4/batch_normalization_18/ReadVariableOp_14sequential_4/batch_normalization_18/ReadVariableOp_12�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_19/ReadVariableOp2sequential_4/batch_normalization_19/ReadVariableOp2l
4sequential_4/batch_normalization_19/ReadVariableOp_14sequential_4/batch_normalization_19/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1762389

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765254

inputs<
.batch_normalization_16_readvariableop_resource:>
0batch_normalization_16_readvariableop_1_resource:M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_16_conv2d_readvariableop_resource: 7
)conv2d_16_biasadd_readvariableop_resource: <
.batch_normalization_17_readvariableop_resource: >
0batch_normalization_17_readvariableop_1_resource: M
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: C
(conv2d_17_conv2d_readvariableop_resource: �8
)conv2d_17_biasadd_readvariableop_resource:	�=
.batch_normalization_18_readvariableop_resource:	�?
0batch_normalization_18_readvariableop_1_resource:	�N
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_18_conv2d_readvariableop_resource:��8
)conv2d_18_biasadd_readvariableop_resource:	�=
.batch_normalization_19_readvariableop_resource:	�?
0batch_normalization_19_readvariableop_1_resource:	�N
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_19_conv2d_readvariableop_resource:��8
)conv2d_19_biasadd_readvariableop_resource:	�<
'dense_12_matmul_readvariableop_resource:��*�7
(dense_12_biasadd_readvariableop_resource:	�;
'dense_13_matmul_readvariableop_resource:
��7
(dense_13_biasadd_readvariableop_resource:	�
identity��6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_16/ReadVariableOp�'batch_normalization_16/ReadVariableOp_1�6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_17/ReadVariableOp�'batch_normalization_17/ReadVariableOp_1�6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_18/ReadVariableOp�'batch_normalization_18/ReadVariableOp_1�6batch_normalization_19/FusedBatchNormV3/ReadVariableOp�8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_19/ReadVariableOp�'batch_normalization_19/ReadVariableOp_1� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�2conv2d_16/kernel/Regularizer/Square/ReadVariableOp� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stack�
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1�
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2�
lambda_4/strided_sliceStridedSliceinputs%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_4/strided_slice�
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp�
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1�
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_16/Conv2D/ReadVariableOp�
conv2d_16/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_16/Conv2D�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_16/Relu�
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_17/ReadVariableOp�
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_17/ReadVariableOp_1�
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_17/Conv2D/ReadVariableOp�
conv2d_17/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
conv2d_17/Conv2D�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
conv2d_17/Relu�
max_pooling2d_4/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool�
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_18/ReadVariableOp�
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_18/ReadVariableOp_1�
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_18/FusedBatchNormV3�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_18/Conv2D�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_18/BiasAdd
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_18/Relu�
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_19/ReadVariableOp�
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_19/ReadVariableOp_1�
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_18/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_19/FusedBatchNormV3�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2DConv2D+batch_normalization_19/FusedBatchNormV3:y:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_19/Conv2D�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
flatten_4/Const�
flatten_4/ReshapeReshapeconv2d_19/Relu:activations:0flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2
flatten_4/Reshape�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_12/Relu�
dropout_8/IdentityIdentitydense_12/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_8/Identity�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMuldropout_8/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_13/Relu�
dropout_9/IdentityIdentitydense_13/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_9/Identity�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentitydropout_9/Identity:output:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp3^conv2d_16/kernel/Regularizer/Square/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766169

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
�
+__inference_conv2d_18_layer_call_fn_1766403

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_18_layer_call_and_return_conditional_losses_17625992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_19_layer_call_fn_1766527

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17629102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
ӟ
�$
A__inference_CNN3_layer_call_and_return_conditional_losses_1764847
input_1I
;sequential_4_batch_normalization_16_readvariableop_resource:K
=sequential_4_batch_normalization_16_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_16_conv2d_readvariableop_resource: D
6sequential_4_conv2d_16_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_17_readvariableop_resource: K
=sequential_4_batch_normalization_17_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_4_conv2d_17_conv2d_readvariableop_resource: �E
6sequential_4_conv2d_17_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_18_readvariableop_resource:	�L
=sequential_4_batch_normalization_18_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_18_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_18_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_19_readvariableop_resource:	�L
=sequential_4_batch_normalization_19_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_19_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_19_biasadd_readvariableop_resource:	�I
4sequential_4_dense_12_matmul_readvariableop_resource:��*�D
5sequential_4_dense_12_biasadd_readvariableop_resource:	�H
4sequential_4_dense_13_matmul_readvariableop_resource:
��D
5sequential_4_dense_13_biasadd_readvariableop_resource:	�:
'dense_14_matmul_readvariableop_resource:	�6
(dense_14_biasadd_readvariableop_resource:
identity��2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�2sequential_4/batch_normalization_16/AssignNewValue�4sequential_4/batch_normalization_16/AssignNewValue_1�Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_16/ReadVariableOp�4sequential_4/batch_normalization_16/ReadVariableOp_1�2sequential_4/batch_normalization_17/AssignNewValue�4sequential_4/batch_normalization_17/AssignNewValue_1�Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_17/ReadVariableOp�4sequential_4/batch_normalization_17/ReadVariableOp_1�2sequential_4/batch_normalization_18/AssignNewValue�4sequential_4/batch_normalization_18/AssignNewValue_1�Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_18/ReadVariableOp�4sequential_4/batch_normalization_18/ReadVariableOp_1�2sequential_4/batch_normalization_19/AssignNewValue�4sequential_4/batch_normalization_19/AssignNewValue_1�Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_19/ReadVariableOp�4sequential_4/batch_normalization_19/ReadVariableOp_1�-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�,sequential_4/conv2d_16/Conv2D/ReadVariableOp�-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�,sequential_4/conv2d_17/Conv2D/ReadVariableOp�-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�,sequential_4/conv2d_18/Conv2D/ReadVariableOp�-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�,sequential_4/conv2d_19/Conv2D/ReadVariableOp�,sequential_4/dense_12/BiasAdd/ReadVariableOp�+sequential_4/dense_12/MatMul/ReadVariableOp�,sequential_4/dense_13/BiasAdd/ReadVariableOp�+sequential_4/dense_13/MatMul/ReadVariableOp�
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack�
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1�
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2�
#sequential_4/lambda_4/strided_sliceStridedSliceinput_12sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_slice�
2sequential_4/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_4/batch_normalization_16/ReadVariableOp�
4sequential_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_4/batch_normalization_16/ReadVariableOp_1�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:0:sequential_4/batch_normalization_16/ReadVariableOp:value:0<sequential_4/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_4/batch_normalization_16/FusedBatchNormV3�
2sequential_4/batch_normalization_16/AssignNewValueAssignVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resourceAsequential_4/batch_normalization_16/FusedBatchNormV3:batch_mean:0D^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_4/batch_normalization_16/AssignNewValue�
4sequential_4/batch_normalization_16/AssignNewValue_1AssignVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resourceEsequential_4/batch_normalization_16/FusedBatchNormV3:batch_variance:0F^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_4/batch_normalization_16/AssignNewValue_1�
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_16/Conv2D/ReadVariableOp�
sequential_4/conv2d_16/Conv2DConv2D8sequential_4/batch_normalization_16/FusedBatchNormV3:y:04sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_4/conv2d_16/Conv2D�
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_4/conv2d_16/BiasAdd�
sequential_4/conv2d_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_4/conv2d_16/Relu�
2sequential_4/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_4/batch_normalization_17/ReadVariableOp�
4sequential_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_4/batch_normalization_17/ReadVariableOp_1�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_16/Relu:activations:0:sequential_4/batch_normalization_17/ReadVariableOp:value:0<sequential_4/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_4/batch_normalization_17/FusedBatchNormV3�
2sequential_4/batch_normalization_17/AssignNewValueAssignVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resourceAsequential_4/batch_normalization_17/FusedBatchNormV3:batch_mean:0D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_4/batch_normalization_17/AssignNewValue�
4sequential_4/batch_normalization_17/AssignNewValue_1AssignVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resourceEsequential_4/batch_normalization_17/FusedBatchNormV3:batch_variance:0F^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_4/batch_normalization_17/AssignNewValue_1�
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_4/conv2d_17/Conv2D/ReadVariableOp�
sequential_4/conv2d_17/Conv2DConv2D8sequential_4/batch_normalization_17/FusedBatchNormV3:y:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_4/conv2d_17/Conv2D�
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_4/conv2d_17/BiasAdd�
sequential_4/conv2d_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_4/conv2d_17/Relu�
$sequential_4/max_pooling2d_4/MaxPoolMaxPool)sequential_4/conv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_4/MaxPool�
2sequential_4/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_18/ReadVariableOp�
4sequential_4/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_18/ReadVariableOp_1�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_4/max_pooling2d_4/MaxPool:output:0:sequential_4/batch_normalization_18/ReadVariableOp:value:0<sequential_4/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_4/batch_normalization_18/FusedBatchNormV3�
2sequential_4/batch_normalization_18/AssignNewValueAssignVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resourceAsequential_4/batch_normalization_18/FusedBatchNormV3:batch_mean:0D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_4/batch_normalization_18/AssignNewValue�
4sequential_4/batch_normalization_18/AssignNewValue_1AssignVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resourceEsequential_4/batch_normalization_18/FusedBatchNormV3:batch_variance:0F^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_4/batch_normalization_18/AssignNewValue_1�
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_18/Conv2D/ReadVariableOp�
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_18/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_18/Conv2D�
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_18/BiasAdd�
sequential_4/conv2d_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_18/Relu�
2sequential_4/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_19/ReadVariableOp�
4sequential_4/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_19/ReadVariableOp_1�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_18/Relu:activations:0:sequential_4/batch_normalization_19/ReadVariableOp:value:0<sequential_4/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_4/batch_normalization_19/FusedBatchNormV3�
2sequential_4/batch_normalization_19/AssignNewValueAssignVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resourceAsequential_4/batch_normalization_19/FusedBatchNormV3:batch_mean:0D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_4/batch_normalization_19/AssignNewValue�
4sequential_4/batch_normalization_19/AssignNewValue_1AssignVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resourceEsequential_4/batch_normalization_19/FusedBatchNormV3:batch_variance:0F^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_4/batch_normalization_19/AssignNewValue_1�
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_19/Conv2D/ReadVariableOp�
sequential_4/conv2d_19/Conv2DConv2D8sequential_4/batch_normalization_19/FusedBatchNormV3:y:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_19/Conv2D�
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_19/BiasAdd�
sequential_4/conv2d_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_19/Relu�
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_4/flatten_4/Const�
sequential_4/flatten_4/ReshapeReshape)sequential_4/conv2d_19/Relu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_4/flatten_4/Reshape�
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp�
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/MatMul�
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp�
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/BiasAdd�
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/Relu�
$sequential_4/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_4/dropout_8/dropout/Const�
"sequential_4/dropout_8/dropout/MulMul(sequential_4/dense_12/Relu:activations:0-sequential_4/dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:����������2$
"sequential_4/dropout_8/dropout/Mul�
$sequential_4/dropout_8/dropout/ShapeShape(sequential_4/dense_12/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_4/dropout_8/dropout/Shape�
;sequential_4/dropout_8/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02=
;sequential_4/dropout_8/dropout/random_uniform/RandomUniform�
-sequential_4/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_4/dropout_8/dropout/GreaterEqual/y�
+sequential_4/dropout_8/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_8/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2-
+sequential_4/dropout_8/dropout/GreaterEqual�
#sequential_4/dropout_8/dropout/CastCast/sequential_4/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2%
#sequential_4/dropout_8/dropout/Cast�
$sequential_4/dropout_8/dropout/Mul_1Mul&sequential_4/dropout_8/dropout/Mul:z:0'sequential_4/dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2&
$sequential_4/dropout_8/dropout/Mul_1�
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp�
sequential_4/dense_13/MatMulMatMul(sequential_4/dropout_8/dropout/Mul_1:z:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/MatMul�
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp�
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/BiasAdd�
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/Relu�
$sequential_4/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_4/dropout_9/dropout/Const�
"sequential_4/dropout_9/dropout/MulMul(sequential_4/dense_13/Relu:activations:0-sequential_4/dropout_9/dropout/Const:output:0*
T0*(
_output_shapes
:����������2$
"sequential_4/dropout_9/dropout/Mul�
$sequential_4/dropout_9/dropout/ShapeShape(sequential_4/dense_13/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_4/dropout_9/dropout/Shape�
;sequential_4/dropout_9/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_9/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02=
;sequential_4/dropout_9/dropout/random_uniform/RandomUniform�
-sequential_4/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_4/dropout_9/dropout/GreaterEqual/y�
+sequential_4/dropout_9/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_9/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_9/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2-
+sequential_4/dropout_9/dropout/GreaterEqual�
#sequential_4/dropout_9/dropout/CastCast/sequential_4/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2%
#sequential_4/dropout_9/dropout/Cast�
$sequential_4/dropout_9/dropout/Mul_1Mul&sequential_4/dropout_9/dropout/Mul:z:0'sequential_4/dropout_9/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2&
$sequential_4/dropout_9/dropout/Mul_1�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMul(sequential_4/dropout_9/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_14/Softmax�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_16/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp3^sequential_4/batch_normalization_16/AssignNewValue5^sequential_4/batch_normalization_16/AssignNewValue_1D^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_16/ReadVariableOp5^sequential_4/batch_normalization_16/ReadVariableOp_13^sequential_4/batch_normalization_17/AssignNewValue5^sequential_4/batch_normalization_17/AssignNewValue_1D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_17/ReadVariableOp5^sequential_4/batch_normalization_17/ReadVariableOp_13^sequential_4/batch_normalization_18/AssignNewValue5^sequential_4/batch_normalization_18/AssignNewValue_1D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_18/ReadVariableOp5^sequential_4/batch_normalization_18/ReadVariableOp_13^sequential_4/batch_normalization_19/AssignNewValue5^sequential_4/batch_normalization_19/AssignNewValue_1D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_19/ReadVariableOp5^sequential_4/batch_normalization_19/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2h
2sequential_4/batch_normalization_16/AssignNewValue2sequential_4/batch_normalization_16/AssignNewValue2l
4sequential_4/batch_normalization_16/AssignNewValue_14sequential_4/batch_normalization_16/AssignNewValue_12�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_16/ReadVariableOp2sequential_4/batch_normalization_16/ReadVariableOp2l
4sequential_4/batch_normalization_16/ReadVariableOp_14sequential_4/batch_normalization_16/ReadVariableOp_12h
2sequential_4/batch_normalization_17/AssignNewValue2sequential_4/batch_normalization_17/AssignNewValue2l
4sequential_4/batch_normalization_17/AssignNewValue_14sequential_4/batch_normalization_17/AssignNewValue_12�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_17/ReadVariableOp2sequential_4/batch_normalization_17/ReadVariableOp2l
4sequential_4/batch_normalization_17/ReadVariableOp_14sequential_4/batch_normalization_17/ReadVariableOp_12h
2sequential_4/batch_normalization_18/AssignNewValue2sequential_4/batch_normalization_18/AssignNewValue2l
4sequential_4/batch_normalization_18/AssignNewValue_14sequential_4/batch_normalization_18/AssignNewValue_12�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_18/ReadVariableOp2sequential_4/batch_normalization_18/ReadVariableOp2l
4sequential_4/batch_normalization_18/ReadVariableOp_14sequential_4/batch_normalization_18/ReadVariableOp_12h
2sequential_4/batch_normalization_19/AssignNewValue2sequential_4/batch_normalization_19/AssignNewValue2l
4sequential_4/batch_normalization_19/AssignNewValue_14sequential_4/batch_normalization_19/AssignNewValue_12�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_19/ReadVariableOp2sequential_4/batch_normalization_19/ReadVariableOp2l
4sequential_4/batch_normalization_19/ReadVariableOp_14sequential_4/batch_normalization_19/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
8__inference_batch_normalization_16_layer_call_fn_1766083

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17630722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
.__inference_sequential_4_layer_call_fn_1765852

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*6
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_17632582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
F__inference_conv2d_18_layer_call_and_return_conditional_losses_1766394

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
G
+__inference_flatten_4_layer_call_fn_1766558

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_17626552
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_1766553

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������*2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�o
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1762736

inputs,
batch_normalization_16_1762484:,
batch_normalization_16_1762486:,
batch_normalization_16_1762488:,
batch_normalization_16_1762490:+
conv2d_16_1762511: 
conv2d_16_1762513: ,
batch_normalization_17_1762534: ,
batch_normalization_17_1762536: ,
batch_normalization_17_1762538: ,
batch_normalization_17_1762540: ,
conv2d_17_1762555: � 
conv2d_17_1762557:	�-
batch_normalization_18_1762579:	�-
batch_normalization_18_1762581:	�-
batch_normalization_18_1762583:	�-
batch_normalization_18_1762585:	�-
conv2d_18_1762600:�� 
conv2d_18_1762602:	�-
batch_normalization_19_1762623:	�-
batch_normalization_19_1762625:	�-
batch_normalization_19_1762627:	�-
batch_normalization_19_1762629:	�-
conv2d_19_1762644:�� 
conv2d_19_1762646:	�%
dense_12_1762675:��*�
dense_12_1762677:	�$
dense_13_1762705:
��
dense_13_1762707:	�
identity��.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall�2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall� dense_12/StatefulPartitionedCall�1dense_12/kernel/Regularizer/Square/ReadVariableOp� dense_13/StatefulPartitionedCall�1dense_13/kernel/Regularizer/Square/ReadVariableOp�
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_17624642
lambda_4/PartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0batch_normalization_16_1762484batch_normalization_16_1762486batch_normalization_16_1762488batch_normalization_16_1762490*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_176248320
.batch_normalization_16/StatefulPartitionedCall�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_16_1762511conv2d_16_1762513*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_16_layer_call_and_return_conditional_losses_17625102#
!conv2d_16/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_17_1762534batch_normalization_17_1762536batch_normalization_17_1762538batch_normalization_17_1762540*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_176253320
.batch_normalization_17/StatefulPartitionedCall�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0conv2d_17_1762555conv2d_17_1762557*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������KK�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_17625542#
!conv2d_17/StatefulPartitionedCall�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17621912!
max_pooling2d_4/PartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_18_1762579batch_normalization_18_1762581batch_normalization_18_1762583batch_normalization_18_1762585*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_176257820
.batch_normalization_18/StatefulPartitionedCall�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_18_1762600conv2d_18_1762602*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_18_layer_call_and_return_conditional_losses_17625992#
!conv2d_18/StatefulPartitionedCall�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_19_1762623batch_normalization_19_1762625batch_normalization_19_1762627batch_normalization_19_1762629*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_176262220
.batch_normalization_19/StatefulPartitionedCall�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0conv2d_19_1762644conv2d_19_1762646*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_19_layer_call_and_return_conditional_losses_17626432#
!conv2d_19/StatefulPartitionedCall�
flatten_4/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_17626552
flatten_4/PartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_1762675dense_12_1762677*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_17626742"
 dense_12/StatefulPartitionedCall�
dropout_8/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_17626852
dropout_8/PartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_13_1762705dense_13_1762707*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_17627042"
 dense_13/StatefulPartitionedCall�
dropout_9/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_17627152
dropout_9/PartitionedCall�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_1762511*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_1762675*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_1762705* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentity"dropout_9/PartitionedCall:output:0/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall3^conv2d_16/kernel/Regularizer/Square/ReadVariableOp"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�9
�
A__inference_CNN3_layer_call_and_return_conditional_losses_1763664

inputs"
sequential_4_1763571:"
sequential_4_1763573:"
sequential_4_1763575:"
sequential_4_1763577:.
sequential_4_1763579: "
sequential_4_1763581: "
sequential_4_1763583: "
sequential_4_1763585: "
sequential_4_1763587: "
sequential_4_1763589: /
sequential_4_1763591: �#
sequential_4_1763593:	�#
sequential_4_1763595:	�#
sequential_4_1763597:	�#
sequential_4_1763599:	�#
sequential_4_1763601:	�0
sequential_4_1763603:��#
sequential_4_1763605:	�#
sequential_4_1763607:	�#
sequential_4_1763609:	�#
sequential_4_1763611:	�#
sequential_4_1763613:	�0
sequential_4_1763615:��#
sequential_4_1763617:	�)
sequential_4_1763619:��*�#
sequential_4_1763621:	�(
sequential_4_1763623:
��#
sequential_4_1763625:	�#
dense_14_1763640:	�
dense_14_1763642:
identity��2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp� dense_14/StatefulPartitionedCall�$sequential_4/StatefulPartitionedCall�
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_1763571sequential_4_1763573sequential_4_1763575sequential_4_1763577sequential_4_1763579sequential_4_1763581sequential_4_1763583sequential_4_1763585sequential_4_1763587sequential_4_1763589sequential_4_1763591sequential_4_1763593sequential_4_1763595sequential_4_1763597sequential_4_1763599sequential_4_1763601sequential_4_1763603sequential_4_1763605sequential_4_1763607sequential_4_1763609sequential_4_1763611sequential_4_1763613sequential_4_1763615sequential_4_1763617sequential_4_1763619sequential_4_1763621sequential_4_1763623sequential_4_1763625*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_17627362&
$sequential_4/StatefulPartitionedCall�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0dense_14_1763640dense_14_1763642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_17636392"
 dense_14/StatefulPartitionedCall�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_1763579*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_1763619*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_1763623* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentity)dense_14/StatefulPartitionedCall:output:03^conv2d_16/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1762483

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_17_layer_call_fn_1766200

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17620812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_1763099

inputs
identity�
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������KK:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1762964

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1766106

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
Relu�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_16/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1763018

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
�
&__inference_CNN3_layer_call_fn_1764912
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_17636642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
e
F__inference_dropout_8_layer_call_and_return_conditional_losses_1762848

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_1766595

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766331

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766457

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1764275
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_17619332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
��
�
__inference_call_1535473

inputsI
;sequential_4_batch_normalization_16_readvariableop_resource:K
=sequential_4_batch_normalization_16_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_16_conv2d_readvariableop_resource: D
6sequential_4_conv2d_16_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_17_readvariableop_resource: K
=sequential_4_batch_normalization_17_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_4_conv2d_17_conv2d_readvariableop_resource: �E
6sequential_4_conv2d_17_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_18_readvariableop_resource:	�L
=sequential_4_batch_normalization_18_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_18_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_18_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_19_readvariableop_resource:	�L
=sequential_4_batch_normalization_19_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_19_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_19_biasadd_readvariableop_resource:	�I
4sequential_4_dense_12_matmul_readvariableop_resource:��*�D
5sequential_4_dense_12_biasadd_readvariableop_resource:	�H
4sequential_4_dense_13_matmul_readvariableop_resource:
��D
5sequential_4_dense_13_biasadd_readvariableop_resource:	�:
'dense_14_matmul_readvariableop_resource:	�6
(dense_14_biasadd_readvariableop_resource:
identity��dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_16/ReadVariableOp�4sequential_4/batch_normalization_16/ReadVariableOp_1�Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_17/ReadVariableOp�4sequential_4/batch_normalization_17/ReadVariableOp_1�Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_18/ReadVariableOp�4sequential_4/batch_normalization_18/ReadVariableOp_1�Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_19/ReadVariableOp�4sequential_4/batch_normalization_19/ReadVariableOp_1�-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�,sequential_4/conv2d_16/Conv2D/ReadVariableOp�-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�,sequential_4/conv2d_17/Conv2D/ReadVariableOp�-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�,sequential_4/conv2d_18/Conv2D/ReadVariableOp�-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�,sequential_4/conv2d_19/Conv2D/ReadVariableOp�,sequential_4/dense_12/BiasAdd/ReadVariableOp�+sequential_4/dense_12/MatMul/ReadVariableOp�,sequential_4/dense_13/BiasAdd/ReadVariableOp�+sequential_4/dense_13/MatMul/ReadVariableOp�
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack�
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1�
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2�
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_slice�
2sequential_4/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_4/batch_normalization_16/ReadVariableOp�
4sequential_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_4/batch_normalization_16/ReadVariableOp_1�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:0:sequential_4/batch_normalization_16/ReadVariableOp:value:0<sequential_4/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_16/FusedBatchNormV3�
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_16/Conv2D/ReadVariableOp�
sequential_4/conv2d_16/Conv2DConv2D8sequential_4/batch_normalization_16/FusedBatchNormV3:y:04sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_4/conv2d_16/Conv2D�
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_4/conv2d_16/BiasAdd�
sequential_4/conv2d_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_4/conv2d_16/Relu�
2sequential_4/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_4/batch_normalization_17/ReadVariableOp�
4sequential_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_4/batch_normalization_17/ReadVariableOp_1�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_16/Relu:activations:0:sequential_4/batch_normalization_17/ReadVariableOp:value:0<sequential_4/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_17/FusedBatchNormV3�
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_4/conv2d_17/Conv2D/ReadVariableOp�
sequential_4/conv2d_17/Conv2DConv2D8sequential_4/batch_normalization_17/FusedBatchNormV3:y:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_4/conv2d_17/Conv2D�
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_4/conv2d_17/BiasAdd�
sequential_4/conv2d_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_4/conv2d_17/Relu�
$sequential_4/max_pooling2d_4/MaxPoolMaxPool)sequential_4/conv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_4/MaxPool�
2sequential_4/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_18/ReadVariableOp�
4sequential_4/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_18/ReadVariableOp_1�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_4/max_pooling2d_4/MaxPool:output:0:sequential_4/batch_normalization_18/ReadVariableOp:value:0<sequential_4/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_18/FusedBatchNormV3�
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_18/Conv2D/ReadVariableOp�
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_18/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_18/Conv2D�
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_18/BiasAdd�
sequential_4/conv2d_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_18/Relu�
2sequential_4/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_19/ReadVariableOp�
4sequential_4/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_19/ReadVariableOp_1�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_18/Relu:activations:0:sequential_4/batch_normalization_19/ReadVariableOp:value:0<sequential_4/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_19/FusedBatchNormV3�
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_19/Conv2D/ReadVariableOp�
sequential_4/conv2d_19/Conv2DConv2D8sequential_4/batch_normalization_19/FusedBatchNormV3:y:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_19/Conv2D�
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_19/BiasAdd�
sequential_4/conv2d_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_19/Relu�
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_4/flatten_4/Const�
sequential_4/flatten_4/ReshapeReshape)sequential_4/conv2d_19/Relu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_4/flatten_4/Reshape�
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp�
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/MatMul�
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp�
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/BiasAdd�
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/Relu�
sequential_4/dropout_8/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:����������2!
sequential_4/dropout_8/Identity�
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp�
sequential_4/dense_13/MatMulMatMul(sequential_4/dropout_8/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/MatMul�
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp�
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/BiasAdd�
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/Relu�
sequential_4/dropout_9/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:����������2!
sequential_4/dropout_9/Identity�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMul(sequential_4/dropout_9/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_14/Softmax�
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpD^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_16/ReadVariableOp5^sequential_4/batch_normalization_16/ReadVariableOp_1D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_17/ReadVariableOp5^sequential_4/batch_normalization_17/ReadVariableOp_1D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_18/ReadVariableOp5^sequential_4/batch_normalization_18/ReadVariableOp_1D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_19/ReadVariableOp5^sequential_4/batch_normalization_19/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_16/ReadVariableOp2sequential_4/batch_normalization_16/ReadVariableOp2l
4sequential_4/batch_normalization_16/ReadVariableOp_14sequential_4/batch_normalization_16/ReadVariableOp_12�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_17/ReadVariableOp2sequential_4/batch_normalization_17/ReadVariableOp2l
4sequential_4/batch_normalization_17/ReadVariableOp_14sequential_4/batch_normalization_17/ReadVariableOp_12�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_18/ReadVariableOp2sequential_4/batch_normalization_18/ReadVariableOp2l
4sequential_4/batch_normalization_18/ReadVariableOp_14sequential_4/batch_normalization_18/ReadVariableOp_12�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_19/ReadVariableOp2sequential_4/batch_normalization_19/ReadVariableOp2l
4sequential_4/batch_normalization_19/ReadVariableOp_14sequential_4/batch_normalization_19/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1766013

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_16_layer_call_fn_1766057

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17619992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
*__inference_dense_12_layer_call_fn_1766590

inputs
unknown:��*�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_17626742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������*: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������*
 
_user_specified_nameinputs
�
�
E__inference_dense_12_layer_call_and_return_conditional_losses_1766581

inputs3
matmul_readvariableop_resource:��*�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������*
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1765995

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1762910

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
E__inference_dense_12_layer_call_and_return_conditional_losses_1762674

inputs3
matmul_readvariableop_resource:��*�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������*
 
_user_specified_nameinputs
�

�
E__inference_dense_14_layer_call_and_return_conditional_losses_1763639

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_4_layer_call_fn_1765791

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_17627362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
П
�$
A__inference_CNN3_layer_call_and_return_conditional_losses_1764561

inputsI
;sequential_4_batch_normalization_16_readvariableop_resource:K
=sequential_4_batch_normalization_16_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_16_conv2d_readvariableop_resource: D
6sequential_4_conv2d_16_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_17_readvariableop_resource: K
=sequential_4_batch_normalization_17_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_4_conv2d_17_conv2d_readvariableop_resource: �E
6sequential_4_conv2d_17_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_18_readvariableop_resource:	�L
=sequential_4_batch_normalization_18_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_18_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_18_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_19_readvariableop_resource:	�L
=sequential_4_batch_normalization_19_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_19_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_19_biasadd_readvariableop_resource:	�I
4sequential_4_dense_12_matmul_readvariableop_resource:��*�D
5sequential_4_dense_12_biasadd_readvariableop_resource:	�H
4sequential_4_dense_13_matmul_readvariableop_resource:
��D
5sequential_4_dense_13_biasadd_readvariableop_resource:	�:
'dense_14_matmul_readvariableop_resource:	�6
(dense_14_biasadd_readvariableop_resource:
identity��2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�2sequential_4/batch_normalization_16/AssignNewValue�4sequential_4/batch_normalization_16/AssignNewValue_1�Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_16/ReadVariableOp�4sequential_4/batch_normalization_16/ReadVariableOp_1�2sequential_4/batch_normalization_17/AssignNewValue�4sequential_4/batch_normalization_17/AssignNewValue_1�Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_17/ReadVariableOp�4sequential_4/batch_normalization_17/ReadVariableOp_1�2sequential_4/batch_normalization_18/AssignNewValue�4sequential_4/batch_normalization_18/AssignNewValue_1�Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_18/ReadVariableOp�4sequential_4/batch_normalization_18/ReadVariableOp_1�2sequential_4/batch_normalization_19/AssignNewValue�4sequential_4/batch_normalization_19/AssignNewValue_1�Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_19/ReadVariableOp�4sequential_4/batch_normalization_19/ReadVariableOp_1�-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�,sequential_4/conv2d_16/Conv2D/ReadVariableOp�-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�,sequential_4/conv2d_17/Conv2D/ReadVariableOp�-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�,sequential_4/conv2d_18/Conv2D/ReadVariableOp�-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�,sequential_4/conv2d_19/Conv2D/ReadVariableOp�,sequential_4/dense_12/BiasAdd/ReadVariableOp�+sequential_4/dense_12/MatMul/ReadVariableOp�,sequential_4/dense_13/BiasAdd/ReadVariableOp�+sequential_4/dense_13/MatMul/ReadVariableOp�
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack�
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1�
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2�
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_slice�
2sequential_4/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_4/batch_normalization_16/ReadVariableOp�
4sequential_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_4/batch_normalization_16/ReadVariableOp_1�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:0:sequential_4/batch_normalization_16/ReadVariableOp:value:0<sequential_4/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_4/batch_normalization_16/FusedBatchNormV3�
2sequential_4/batch_normalization_16/AssignNewValueAssignVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resourceAsequential_4/batch_normalization_16/FusedBatchNormV3:batch_mean:0D^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_4/batch_normalization_16/AssignNewValue�
4sequential_4/batch_normalization_16/AssignNewValue_1AssignVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resourceEsequential_4/batch_normalization_16/FusedBatchNormV3:batch_variance:0F^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_4/batch_normalization_16/AssignNewValue_1�
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_16/Conv2D/ReadVariableOp�
sequential_4/conv2d_16/Conv2DConv2D8sequential_4/batch_normalization_16/FusedBatchNormV3:y:04sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_4/conv2d_16/Conv2D�
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_4/conv2d_16/BiasAdd�
sequential_4/conv2d_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_4/conv2d_16/Relu�
2sequential_4/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_4/batch_normalization_17/ReadVariableOp�
4sequential_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_4/batch_normalization_17/ReadVariableOp_1�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_16/Relu:activations:0:sequential_4/batch_normalization_17/ReadVariableOp:value:0<sequential_4/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_4/batch_normalization_17/FusedBatchNormV3�
2sequential_4/batch_normalization_17/AssignNewValueAssignVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resourceAsequential_4/batch_normalization_17/FusedBatchNormV3:batch_mean:0D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_4/batch_normalization_17/AssignNewValue�
4sequential_4/batch_normalization_17/AssignNewValue_1AssignVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resourceEsequential_4/batch_normalization_17/FusedBatchNormV3:batch_variance:0F^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_4/batch_normalization_17/AssignNewValue_1�
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_4/conv2d_17/Conv2D/ReadVariableOp�
sequential_4/conv2d_17/Conv2DConv2D8sequential_4/batch_normalization_17/FusedBatchNormV3:y:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_4/conv2d_17/Conv2D�
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_4/conv2d_17/BiasAdd�
sequential_4/conv2d_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_4/conv2d_17/Relu�
$sequential_4/max_pooling2d_4/MaxPoolMaxPool)sequential_4/conv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_4/MaxPool�
2sequential_4/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_18/ReadVariableOp�
4sequential_4/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_18/ReadVariableOp_1�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_4/max_pooling2d_4/MaxPool:output:0:sequential_4/batch_normalization_18/ReadVariableOp:value:0<sequential_4/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_4/batch_normalization_18/FusedBatchNormV3�
2sequential_4/batch_normalization_18/AssignNewValueAssignVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resourceAsequential_4/batch_normalization_18/FusedBatchNormV3:batch_mean:0D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_4/batch_normalization_18/AssignNewValue�
4sequential_4/batch_normalization_18/AssignNewValue_1AssignVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resourceEsequential_4/batch_normalization_18/FusedBatchNormV3:batch_variance:0F^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_4/batch_normalization_18/AssignNewValue_1�
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_18/Conv2D/ReadVariableOp�
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_18/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_18/Conv2D�
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_18/BiasAdd�
sequential_4/conv2d_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_18/Relu�
2sequential_4/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_19/ReadVariableOp�
4sequential_4/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_19/ReadVariableOp_1�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_18/Relu:activations:0:sequential_4/batch_normalization_19/ReadVariableOp:value:0<sequential_4/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_4/batch_normalization_19/FusedBatchNormV3�
2sequential_4/batch_normalization_19/AssignNewValueAssignVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resourceAsequential_4/batch_normalization_19/FusedBatchNormV3:batch_mean:0D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_4/batch_normalization_19/AssignNewValue�
4sequential_4/batch_normalization_19/AssignNewValue_1AssignVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resourceEsequential_4/batch_normalization_19/FusedBatchNormV3:batch_variance:0F^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_4/batch_normalization_19/AssignNewValue_1�
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_19/Conv2D/ReadVariableOp�
sequential_4/conv2d_19/Conv2DConv2D8sequential_4/batch_normalization_19/FusedBatchNormV3:y:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_19/Conv2D�
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_19/BiasAdd�
sequential_4/conv2d_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_19/Relu�
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_4/flatten_4/Const�
sequential_4/flatten_4/ReshapeReshape)sequential_4/conv2d_19/Relu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_4/flatten_4/Reshape�
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp�
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/MatMul�
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp�
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/BiasAdd�
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/Relu�
$sequential_4/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_4/dropout_8/dropout/Const�
"sequential_4/dropout_8/dropout/MulMul(sequential_4/dense_12/Relu:activations:0-sequential_4/dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:����������2$
"sequential_4/dropout_8/dropout/Mul�
$sequential_4/dropout_8/dropout/ShapeShape(sequential_4/dense_12/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_4/dropout_8/dropout/Shape�
;sequential_4/dropout_8/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02=
;sequential_4/dropout_8/dropout/random_uniform/RandomUniform�
-sequential_4/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_4/dropout_8/dropout/GreaterEqual/y�
+sequential_4/dropout_8/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_8/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2-
+sequential_4/dropout_8/dropout/GreaterEqual�
#sequential_4/dropout_8/dropout/CastCast/sequential_4/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2%
#sequential_4/dropout_8/dropout/Cast�
$sequential_4/dropout_8/dropout/Mul_1Mul&sequential_4/dropout_8/dropout/Mul:z:0'sequential_4/dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2&
$sequential_4/dropout_8/dropout/Mul_1�
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp�
sequential_4/dense_13/MatMulMatMul(sequential_4/dropout_8/dropout/Mul_1:z:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/MatMul�
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp�
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/BiasAdd�
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/Relu�
$sequential_4/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_4/dropout_9/dropout/Const�
"sequential_4/dropout_9/dropout/MulMul(sequential_4/dense_13/Relu:activations:0-sequential_4/dropout_9/dropout/Const:output:0*
T0*(
_output_shapes
:����������2$
"sequential_4/dropout_9/dropout/Mul�
$sequential_4/dropout_9/dropout/ShapeShape(sequential_4/dense_13/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_4/dropout_9/dropout/Shape�
;sequential_4/dropout_9/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_9/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02=
;sequential_4/dropout_9/dropout/random_uniform/RandomUniform�
-sequential_4/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_4/dropout_9/dropout/GreaterEqual/y�
+sequential_4/dropout_9/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_9/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_9/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2-
+sequential_4/dropout_9/dropout/GreaterEqual�
#sequential_4/dropout_9/dropout/CastCast/sequential_4/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2%
#sequential_4/dropout_9/dropout/Cast�
$sequential_4/dropout_9/dropout/Mul_1Mul&sequential_4/dropout_9/dropout/Mul:z:0'sequential_4/dropout_9/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2&
$sequential_4/dropout_9/dropout/Mul_1�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMul(sequential_4/dropout_9/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_14/Softmax�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_16/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp3^sequential_4/batch_normalization_16/AssignNewValue5^sequential_4/batch_normalization_16/AssignNewValue_1D^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_16/ReadVariableOp5^sequential_4/batch_normalization_16/ReadVariableOp_13^sequential_4/batch_normalization_17/AssignNewValue5^sequential_4/batch_normalization_17/AssignNewValue_1D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_17/ReadVariableOp5^sequential_4/batch_normalization_17/ReadVariableOp_13^sequential_4/batch_normalization_18/AssignNewValue5^sequential_4/batch_normalization_18/AssignNewValue_1D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_18/ReadVariableOp5^sequential_4/batch_normalization_18/ReadVariableOp_13^sequential_4/batch_normalization_19/AssignNewValue5^sequential_4/batch_normalization_19/AssignNewValue_1D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_19/ReadVariableOp5^sequential_4/batch_normalization_19/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2h
2sequential_4/batch_normalization_16/AssignNewValue2sequential_4/batch_normalization_16/AssignNewValue2l
4sequential_4/batch_normalization_16/AssignNewValue_14sequential_4/batch_normalization_16/AssignNewValue_12�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_16/ReadVariableOp2sequential_4/batch_normalization_16/ReadVariableOp2l
4sequential_4/batch_normalization_16/ReadVariableOp_14sequential_4/batch_normalization_16/ReadVariableOp_12h
2sequential_4/batch_normalization_17/AssignNewValue2sequential_4/batch_normalization_17/AssignNewValue2l
4sequential_4/batch_normalization_17/AssignNewValue_14sequential_4/batch_normalization_17/AssignNewValue_12�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_17/ReadVariableOp2sequential_4/batch_normalization_17/ReadVariableOp2l
4sequential_4/batch_normalization_17/ReadVariableOp_14sequential_4/batch_normalization_17/ReadVariableOp_12h
2sequential_4/batch_normalization_18/AssignNewValue2sequential_4/batch_normalization_18/AssignNewValue2l
4sequential_4/batch_normalization_18/AssignNewValue_14sequential_4/batch_normalization_18/AssignNewValue_12�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_18/ReadVariableOp2sequential_4/batch_normalization_18/ReadVariableOp2l
4sequential_4/batch_normalization_18/ReadVariableOp_14sequential_4/batch_normalization_18/ReadVariableOp_12h
2sequential_4/batch_normalization_19/AssignNewValue2sequential_4/batch_normalization_19/AssignNewValue2l
4sequential_4/batch_normalization_19/AssignNewValue_14sequential_4/batch_normalization_19/AssignNewValue_12�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_19/ReadVariableOp2sequential_4/batch_normalization_19/ReadVariableOp2l
4sequential_4/batch_normalization_19/ReadVariableOp_14sequential_4/batch_normalization_19/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
F
*__inference_lambda_4_layer_call_fn_1765959

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
:���������KK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_17630992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������KK:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
F__inference_conv2d_19_layer_call_and_return_conditional_losses_1762643

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766313

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1762533

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
e
F__inference_dropout_9_layer_call_and_return_conditional_losses_1762815

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�!
A__inference_CNN3_layer_call_and_return_conditional_losses_1764411

inputsI
;sequential_4_batch_normalization_16_readvariableop_resource:K
=sequential_4_batch_normalization_16_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_16_conv2d_readvariableop_resource: D
6sequential_4_conv2d_16_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_17_readvariableop_resource: K
=sequential_4_batch_normalization_17_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_4_conv2d_17_conv2d_readvariableop_resource: �E
6sequential_4_conv2d_17_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_18_readvariableop_resource:	�L
=sequential_4_batch_normalization_18_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_18_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_18_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_19_readvariableop_resource:	�L
=sequential_4_batch_normalization_19_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_19_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_19_biasadd_readvariableop_resource:	�I
4sequential_4_dense_12_matmul_readvariableop_resource:��*�D
5sequential_4_dense_12_biasadd_readvariableop_resource:	�H
4sequential_4_dense_13_matmul_readvariableop_resource:
��D
5sequential_4_dense_13_biasadd_readvariableop_resource:	�:
'dense_14_matmul_readvariableop_resource:	�6
(dense_14_biasadd_readvariableop_resource:
identity��2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_16/ReadVariableOp�4sequential_4/batch_normalization_16/ReadVariableOp_1�Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_17/ReadVariableOp�4sequential_4/batch_normalization_17/ReadVariableOp_1�Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_18/ReadVariableOp�4sequential_4/batch_normalization_18/ReadVariableOp_1�Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_19/ReadVariableOp�4sequential_4/batch_normalization_19/ReadVariableOp_1�-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�,sequential_4/conv2d_16/Conv2D/ReadVariableOp�-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�,sequential_4/conv2d_17/Conv2D/ReadVariableOp�-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�,sequential_4/conv2d_18/Conv2D/ReadVariableOp�-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�,sequential_4/conv2d_19/Conv2D/ReadVariableOp�,sequential_4/dense_12/BiasAdd/ReadVariableOp�+sequential_4/dense_12/MatMul/ReadVariableOp�,sequential_4/dense_13/BiasAdd/ReadVariableOp�+sequential_4/dense_13/MatMul/ReadVariableOp�
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack�
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1�
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2�
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_slice�
2sequential_4/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_4/batch_normalization_16/ReadVariableOp�
4sequential_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_4/batch_normalization_16/ReadVariableOp_1�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:0:sequential_4/batch_normalization_16/ReadVariableOp:value:0<sequential_4/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_16/FusedBatchNormV3�
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_16/Conv2D/ReadVariableOp�
sequential_4/conv2d_16/Conv2DConv2D8sequential_4/batch_normalization_16/FusedBatchNormV3:y:04sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_4/conv2d_16/Conv2D�
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_4/conv2d_16/BiasAdd�
sequential_4/conv2d_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_4/conv2d_16/Relu�
2sequential_4/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_4/batch_normalization_17/ReadVariableOp�
4sequential_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_4/batch_normalization_17/ReadVariableOp_1�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_16/Relu:activations:0:sequential_4/batch_normalization_17/ReadVariableOp:value:0<sequential_4/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_17/FusedBatchNormV3�
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_4/conv2d_17/Conv2D/ReadVariableOp�
sequential_4/conv2d_17/Conv2DConv2D8sequential_4/batch_normalization_17/FusedBatchNormV3:y:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_4/conv2d_17/Conv2D�
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_4/conv2d_17/BiasAdd�
sequential_4/conv2d_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_4/conv2d_17/Relu�
$sequential_4/max_pooling2d_4/MaxPoolMaxPool)sequential_4/conv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_4/MaxPool�
2sequential_4/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_18/ReadVariableOp�
4sequential_4/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_18/ReadVariableOp_1�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_4/max_pooling2d_4/MaxPool:output:0:sequential_4/batch_normalization_18/ReadVariableOp:value:0<sequential_4/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_18/FusedBatchNormV3�
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_18/Conv2D/ReadVariableOp�
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_18/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_18/Conv2D�
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_18/BiasAdd�
sequential_4/conv2d_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_18/Relu�
2sequential_4/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_19/ReadVariableOp�
4sequential_4/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_19/ReadVariableOp_1�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_18/Relu:activations:0:sequential_4/batch_normalization_19/ReadVariableOp:value:0<sequential_4/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_19/FusedBatchNormV3�
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_19/Conv2D/ReadVariableOp�
sequential_4/conv2d_19/Conv2DConv2D8sequential_4/batch_normalization_19/FusedBatchNormV3:y:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_19/Conv2D�
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_19/BiasAdd�
sequential_4/conv2d_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_19/Relu�
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_4/flatten_4/Const�
sequential_4/flatten_4/ReshapeReshape)sequential_4/conv2d_19/Relu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_4/flatten_4/Reshape�
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp�
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/MatMul�
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp�
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/BiasAdd�
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/Relu�
sequential_4/dropout_8/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:����������2!
sequential_4/dropout_8/Identity�
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp�
sequential_4/dense_13/MatMulMatMul(sequential_4/dropout_8/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/MatMul�
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp�
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/BiasAdd�
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/Relu�
sequential_4/dropout_9/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:����������2!
sequential_4/dropout_9/Identity�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMul(sequential_4/dropout_9/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_14/Softmax�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_16/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpD^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_16/ReadVariableOp5^sequential_4/batch_normalization_16/ReadVariableOp_1D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_17/ReadVariableOp5^sequential_4/batch_normalization_17/ReadVariableOp_1D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_18/ReadVariableOp5^sequential_4/batch_normalization_18/ReadVariableOp_1D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_19/ReadVariableOp5^sequential_4/batch_normalization_19/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_16/ReadVariableOp2sequential_4/batch_normalization_16/ReadVariableOp2l
4sequential_4/batch_normalization_16/ReadVariableOp_14sequential_4/batch_normalization_16/ReadVariableOp_12�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_17/ReadVariableOp2sequential_4/batch_normalization_17/ReadVariableOp2l
4sequential_4/batch_normalization_17/ReadVariableOp_14sequential_4/batch_normalization_17/ReadVariableOp_12�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_18/ReadVariableOp2sequential_4/batch_normalization_18/ReadVariableOp2l
4sequential_4/batch_normalization_18/ReadVariableOp_14sequential_4/batch_normalization_18/ReadVariableOp_12�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_19/ReadVariableOp2sequential_4/batch_normalization_19/ReadVariableOp2l
4sequential_4/batch_normalization_19/ReadVariableOp_14sequential_4/batch_normalization_19/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
d
+__inference_dropout_9_layer_call_fn_1766676

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_17628152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_19_layer_call_fn_1766501

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17623892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766421

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�!
A__inference_CNN3_layer_call_and_return_conditional_losses_1764697
input_1I
;sequential_4_batch_normalization_16_readvariableop_resource:K
=sequential_4_batch_normalization_16_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_16_conv2d_readvariableop_resource: D
6sequential_4_conv2d_16_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_17_readvariableop_resource: K
=sequential_4_batch_normalization_17_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_4_conv2d_17_conv2d_readvariableop_resource: �E
6sequential_4_conv2d_17_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_18_readvariableop_resource:	�L
=sequential_4_batch_normalization_18_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_18_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_18_biasadd_readvariableop_resource:	�J
;sequential_4_batch_normalization_19_readvariableop_resource:	�L
=sequential_4_batch_normalization_19_readvariableop_1_resource:	�[
Lsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_4_conv2d_19_conv2d_readvariableop_resource:��E
6sequential_4_conv2d_19_biasadd_readvariableop_resource:	�I
4sequential_4_dense_12_matmul_readvariableop_resource:��*�D
5sequential_4_dense_12_biasadd_readvariableop_resource:	�H
4sequential_4_dense_13_matmul_readvariableop_resource:
��D
5sequential_4_dense_13_biasadd_readvariableop_resource:	�:
'dense_14_matmul_readvariableop_resource:	�6
(dense_14_biasadd_readvariableop_resource:
identity��2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_16/ReadVariableOp�4sequential_4/batch_normalization_16/ReadVariableOp_1�Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_17/ReadVariableOp�4sequential_4/batch_normalization_17/ReadVariableOp_1�Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_18/ReadVariableOp�4sequential_4/batch_normalization_18/ReadVariableOp_1�Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�2sequential_4/batch_normalization_19/ReadVariableOp�4sequential_4/batch_normalization_19/ReadVariableOp_1�-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�,sequential_4/conv2d_16/Conv2D/ReadVariableOp�-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�,sequential_4/conv2d_17/Conv2D/ReadVariableOp�-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�,sequential_4/conv2d_18/Conv2D/ReadVariableOp�-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�,sequential_4/conv2d_19/Conv2D/ReadVariableOp�,sequential_4/dense_12/BiasAdd/ReadVariableOp�+sequential_4/dense_12/MatMul/ReadVariableOp�,sequential_4/dense_13/BiasAdd/ReadVariableOp�+sequential_4/dense_13/MatMul/ReadVariableOp�
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack�
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1�
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2�
#sequential_4/lambda_4/strided_sliceStridedSliceinput_12sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_slice�
2sequential_4/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_4/batch_normalization_16/ReadVariableOp�
4sequential_4/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_4/batch_normalization_16/ReadVariableOp_1�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:0:sequential_4/batch_normalization_16/ReadVariableOp:value:0<sequential_4/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_16/FusedBatchNormV3�
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_16/Conv2D/ReadVariableOp�
sequential_4/conv2d_16/Conv2DConv2D8sequential_4/batch_normalization_16/FusedBatchNormV3:y:04sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_4/conv2d_16/Conv2D�
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp�
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_4/conv2d_16/BiasAdd�
sequential_4/conv2d_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_4/conv2d_16/Relu�
2sequential_4/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_4/batch_normalization_17/ReadVariableOp�
4sequential_4/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_4/batch_normalization_17/ReadVariableOp_1�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_16/Relu:activations:0:sequential_4/batch_normalization_17/ReadVariableOp:value:0<sequential_4/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_17/FusedBatchNormV3�
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_4/conv2d_17/Conv2D/ReadVariableOp�
sequential_4/conv2d_17/Conv2DConv2D8sequential_4/batch_normalization_17/FusedBatchNormV3:y:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_4/conv2d_17/Conv2D�
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp�
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_4/conv2d_17/BiasAdd�
sequential_4/conv2d_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_4/conv2d_17/Relu�
$sequential_4/max_pooling2d_4/MaxPoolMaxPool)sequential_4/conv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_4/MaxPool�
2sequential_4/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_18/ReadVariableOp�
4sequential_4/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_18/ReadVariableOp_1�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_4/max_pooling2d_4/MaxPool:output:0:sequential_4/batch_normalization_18/ReadVariableOp:value:0<sequential_4/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_18/FusedBatchNormV3�
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_18/Conv2D/ReadVariableOp�
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_18/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_18/Conv2D�
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp�
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_18/BiasAdd�
sequential_4/conv2d_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_18/Relu�
2sequential_4/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_4/batch_normalization_19/ReadVariableOp�
4sequential_4/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_4/batch_normalization_19/ReadVariableOp_1�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
4sequential_4/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_4/conv2d_18/Relu:activations:0:sequential_4/batch_normalization_19/ReadVariableOp:value:0<sequential_4/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_4/batch_normalization_19/FusedBatchNormV3�
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_4/conv2d_19/Conv2D/ReadVariableOp�
sequential_4/conv2d_19/Conv2DConv2D8sequential_4/batch_normalization_19/FusedBatchNormV3:y:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_4/conv2d_19/Conv2D�
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp�
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_4/conv2d_19/BiasAdd�
sequential_4/conv2d_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_4/conv2d_19/Relu�
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_4/flatten_4/Const�
sequential_4/flatten_4/ReshapeReshape)sequential_4/conv2d_19/Relu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_4/flatten_4/Reshape�
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp�
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/MatMul�
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp�
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/BiasAdd�
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_12/Relu�
sequential_4/dropout_8/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:����������2!
sequential_4/dropout_8/Identity�
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp�
sequential_4/dense_13/MatMulMatMul(sequential_4/dropout_8/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/MatMul�
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp�
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/BiasAdd�
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_4/dense_13/Relu�
sequential_4/dropout_9/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:����������2!
sequential_4/dropout_9/Identity�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMul(sequential_4/dropout_9/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_14/Softmax�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_16/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpD^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_16/ReadVariableOp5^sequential_4/batch_normalization_16/ReadVariableOp_1D^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_17/ReadVariableOp5^sequential_4/batch_normalization_17/ReadVariableOp_1D^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_18/ReadVariableOp5^sequential_4/batch_normalization_18/ReadVariableOp_1D^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_19/ReadVariableOp5^sequential_4/batch_normalization_19/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2�
Csequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_16/ReadVariableOp2sequential_4/batch_normalization_16/ReadVariableOp2l
4sequential_4/batch_normalization_16/ReadVariableOp_14sequential_4/batch_normalization_16/ReadVariableOp_12�
Csequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_17/ReadVariableOp2sequential_4/batch_normalization_17/ReadVariableOp2l
4sequential_4/batch_normalization_17/ReadVariableOp_14sequential_4/batch_normalization_17/ReadVariableOp_12�
Csequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_18/ReadVariableOp2sequential_4/batch_normalization_18/ReadVariableOp2l
4sequential_4/batch_normalization_18/ReadVariableOp_14sequential_4/batch_normalization_18/ReadVariableOp_12�
Csequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2�
Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_19/ReadVariableOp2sequential_4/batch_normalization_19/ReadVariableOp2l
4sequential_4/batch_normalization_19/ReadVariableOp_14sequential_4/batch_normalization_19/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1761955

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
.__inference_sequential_4_layer_call_fn_1765913
lambda_4_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: $
	unknown_9: �

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:��*�

unknown_24:	�

unknown_25:
��

unknown_26:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*6
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_17632582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_4_input
��
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765397

inputs<
.batch_normalization_16_readvariableop_resource:>
0batch_normalization_16_readvariableop_1_resource:M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_16_conv2d_readvariableop_resource: 7
)conv2d_16_biasadd_readvariableop_resource: <
.batch_normalization_17_readvariableop_resource: >
0batch_normalization_17_readvariableop_1_resource: M
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource: C
(conv2d_17_conv2d_readvariableop_resource: �8
)conv2d_17_biasadd_readvariableop_resource:	�=
.batch_normalization_18_readvariableop_resource:	�?
0batch_normalization_18_readvariableop_1_resource:	�N
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_18_conv2d_readvariableop_resource:��8
)conv2d_18_biasadd_readvariableop_resource:	�=
.batch_normalization_19_readvariableop_resource:	�?
0batch_normalization_19_readvariableop_1_resource:	�N
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_19_conv2d_readvariableop_resource:��8
)conv2d_19_biasadd_readvariableop_resource:	�<
'dense_12_matmul_readvariableop_resource:��*�7
(dense_12_biasadd_readvariableop_resource:	�;
'dense_13_matmul_readvariableop_resource:
��7
(dense_13_biasadd_readvariableop_resource:	�
identity��%batch_normalization_16/AssignNewValue�'batch_normalization_16/AssignNewValue_1�6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_16/ReadVariableOp�'batch_normalization_16/ReadVariableOp_1�%batch_normalization_17/AssignNewValue�'batch_normalization_17/AssignNewValue_1�6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_17/ReadVariableOp�'batch_normalization_17/ReadVariableOp_1�%batch_normalization_18/AssignNewValue�'batch_normalization_18/AssignNewValue_1�6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_18/ReadVariableOp�'batch_normalization_18/ReadVariableOp_1�%batch_normalization_19/AssignNewValue�'batch_normalization_19/AssignNewValue_1�6batch_normalization_19/FusedBatchNormV3/ReadVariableOp�8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_19/ReadVariableOp�'batch_normalization_19/ReadVariableOp_1� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�2conv2d_16/kernel/Regularizer/Square/ReadVariableOp� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�1dense_12/kernel/Regularizer/Square/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�1dense_13/kernel/Regularizer/Square/ReadVariableOp�
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stack�
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1�
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2�
lambda_4/strided_sliceStridedSliceinputs%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_4/strided_slice�
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp�
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1�
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_16/FusedBatchNormV3�
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValue�
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_16/Conv2D/ReadVariableOp�
conv2d_16/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_16/Conv2D�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_16/Relu�
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_17/ReadVariableOp�
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_17/ReadVariableOp_1�
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_17/FusedBatchNormV3�
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue�
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_17/Conv2D/ReadVariableOp�
conv2d_17/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
conv2d_17/Conv2D�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
conv2d_17/Relu�
max_pooling2d_4/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool�
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_18/ReadVariableOp�
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_18/ReadVariableOp_1�
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_18/FusedBatchNormV3�
%batch_normalization_18/AssignNewValueAssignVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource4batch_normalization_18/FusedBatchNormV3:batch_mean:07^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_18/AssignNewValue�
'batch_normalization_18/AssignNewValue_1AssignVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_18/FusedBatchNormV3:batch_variance:09^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_18/AssignNewValue_1�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_18/Conv2D�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_18/BiasAdd
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_18/Relu�
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_19/ReadVariableOp�
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_19/ReadVariableOp_1�
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_18/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_19/FusedBatchNormV3�
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_19/AssignNewValue�
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_19/AssignNewValue_1�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2DConv2D+batch_normalization_19/FusedBatchNormV3:y:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_19/Conv2D�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
flatten_4/Const�
flatten_4/ReshapeReshapeconv2d_19/Relu:activations:0flatten_4/Const:output:0*
T0*)
_output_shapes
:�����������*2
flatten_4/Reshape�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_12/Reluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_8/dropout/Const�
dropout_8/dropout/MulMuldense_12/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_8/dropout/Mul}
dropout_8/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape�
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform�
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_8/dropout/GreaterEqual/y�
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2 
dropout_8/dropout/GreaterEqual�
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_8/dropout/Cast�
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_8/dropout/Mul_1�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMuldropout_8/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_13/Reluw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_9/dropout/Const�
dropout_9/dropout/MulMuldense_13/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_9/dropout/Mul}
dropout_9/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform�
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_9/dropout/GreaterEqual/y�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2 
dropout_9/dropout/GreaterEqual�
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_9/dropout/Cast�
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_9/dropout/Mul_1�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp�
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_12/kernel/Regularizer/Square�
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/Const�
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/Sum�
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_12/kernel/Regularizer/mul/x�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul�
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOp�
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_13/kernel/Regularizer/Square�
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/Const�
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/Sum�
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_13/kernel/Regularizer/mul/x�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul�
IdentityIdentitydropout_9/dropout/Mul_1:z:0&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1&^batch_normalization_18/AssignNewValue(^batch_normalization_18/AssignNewValue_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp3^conv2d_16/kernel/Regularizer/Square/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12N
%batch_normalization_18/AssignNewValue%batch_normalization_18/AssignNewValue2R
'batch_normalization_18/AssignNewValue_1'batch_normalization_18/AssignNewValue_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1762554

inputs9
conv2d_readvariableop_resource: �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������KK�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1763072

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1762219

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1762510

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
Relu�
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_16/kernel/Regularizer/Square�
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/Const�
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum�
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_16/kernel/Regularizer/mul/x�
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_16/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_16/kernel/Regularizer/Square/ReadVariableOp2conv2d_16/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_17_layer_call_fn_1766239

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17630182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������KK : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
d
+__inference_dropout_8_layer_call_fn_1766617

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_17628482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1762622

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������%%�: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
��
�6
#__inference__traced_restore_1767240
file_prefix3
 assignvariableop_dense_14_kernel:	�.
 assignvariableop_1_dense_14_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: =
/assignvariableop_7_batch_normalization_16_gamma:<
.assignvariableop_8_batch_normalization_16_beta:C
5assignvariableop_9_batch_normalization_16_moving_mean:H
:assignvariableop_10_batch_normalization_16_moving_variance:>
$assignvariableop_11_conv2d_16_kernel: 0
"assignvariableop_12_conv2d_16_bias: >
0assignvariableop_13_batch_normalization_17_gamma: =
/assignvariableop_14_batch_normalization_17_beta: D
6assignvariableop_15_batch_normalization_17_moving_mean: H
:assignvariableop_16_batch_normalization_17_moving_variance: ?
$assignvariableop_17_conv2d_17_kernel: �1
"assignvariableop_18_conv2d_17_bias:	�?
0assignvariableop_19_batch_normalization_18_gamma:	�>
/assignvariableop_20_batch_normalization_18_beta:	�E
6assignvariableop_21_batch_normalization_18_moving_mean:	�I
:assignvariableop_22_batch_normalization_18_moving_variance:	�@
$assignvariableop_23_conv2d_18_kernel:��1
"assignvariableop_24_conv2d_18_bias:	�?
0assignvariableop_25_batch_normalization_19_gamma:	�>
/assignvariableop_26_batch_normalization_19_beta:	�E
6assignvariableop_27_batch_normalization_19_moving_mean:	�I
:assignvariableop_28_batch_normalization_19_moving_variance:	�@
$assignvariableop_29_conv2d_19_kernel:��1
"assignvariableop_30_conv2d_19_bias:	�8
#assignvariableop_31_dense_12_kernel:��*�0
!assignvariableop_32_dense_12_bias:	�7
#assignvariableop_33_dense_13_kernel:
��0
!assignvariableop_34_dense_13_bias:	�#
assignvariableop_35_total: #
assignvariableop_36_count: %
assignvariableop_37_total_1: %
assignvariableop_38_count_1: =
*assignvariableop_39_adam_dense_14_kernel_m:	�6
(assignvariableop_40_adam_dense_14_bias_m:E
7assignvariableop_41_adam_batch_normalization_16_gamma_m:D
6assignvariableop_42_adam_batch_normalization_16_beta_m:E
+assignvariableop_43_adam_conv2d_16_kernel_m: 7
)assignvariableop_44_adam_conv2d_16_bias_m: E
7assignvariableop_45_adam_batch_normalization_17_gamma_m: D
6assignvariableop_46_adam_batch_normalization_17_beta_m: F
+assignvariableop_47_adam_conv2d_17_kernel_m: �8
)assignvariableop_48_adam_conv2d_17_bias_m:	�F
7assignvariableop_49_adam_batch_normalization_18_gamma_m:	�E
6assignvariableop_50_adam_batch_normalization_18_beta_m:	�G
+assignvariableop_51_adam_conv2d_18_kernel_m:��8
)assignvariableop_52_adam_conv2d_18_bias_m:	�F
7assignvariableop_53_adam_batch_normalization_19_gamma_m:	�E
6assignvariableop_54_adam_batch_normalization_19_beta_m:	�G
+assignvariableop_55_adam_conv2d_19_kernel_m:��8
)assignvariableop_56_adam_conv2d_19_bias_m:	�?
*assignvariableop_57_adam_dense_12_kernel_m:��*�7
(assignvariableop_58_adam_dense_12_bias_m:	�>
*assignvariableop_59_adam_dense_13_kernel_m:
��7
(assignvariableop_60_adam_dense_13_bias_m:	�=
*assignvariableop_61_adam_dense_14_kernel_v:	�6
(assignvariableop_62_adam_dense_14_bias_v:E
7assignvariableop_63_adam_batch_normalization_16_gamma_v:D
6assignvariableop_64_adam_batch_normalization_16_beta_v:E
+assignvariableop_65_adam_conv2d_16_kernel_v: 7
)assignvariableop_66_adam_conv2d_16_bias_v: E
7assignvariableop_67_adam_batch_normalization_17_gamma_v: D
6assignvariableop_68_adam_batch_normalization_17_beta_v: F
+assignvariableop_69_adam_conv2d_17_kernel_v: �8
)assignvariableop_70_adam_conv2d_17_bias_v:	�F
7assignvariableop_71_adam_batch_normalization_18_gamma_v:	�E
6assignvariableop_72_adam_batch_normalization_18_beta_v:	�G
+assignvariableop_73_adam_conv2d_18_kernel_v:��8
)assignvariableop_74_adam_conv2d_18_bias_v:	�F
7assignvariableop_75_adam_batch_normalization_19_gamma_v:	�E
6assignvariableop_76_adam_batch_normalization_19_beta_v:	�G
+assignvariableop_77_adam_conv2d_19_kernel_v:��8
)assignvariableop_78_adam_conv2d_19_bias_v:	�?
*assignvariableop_79_adam_dense_12_kernel_v:��*�7
(assignvariableop_80_adam_dense_12_bias_v:	�>
*assignvariableop_81_adam_dense_13_kernel_v:
��7
(assignvariableop_82_adam_dense_13_bias_v:	�
identity_84��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_9�%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�%
value�%B�$TB)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�
value�B�TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_16_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_16_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp5assignvariableop_9_batch_normalization_16_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp:assignvariableop_10_batch_normalization_16_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_16_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_16_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_17_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_17_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp6assignvariableop_15_batch_normalization_17_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp:assignvariableop_16_batch_normalization_17_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_17_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_17_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp0assignvariableop_19_batch_normalization_18_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_18_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_batch_normalization_18_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp:assignvariableop_22_batch_normalization_18_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp$assignvariableop_23_conv2d_18_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_conv2d_18_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp0assignvariableop_25_batch_normalization_19_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_19_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_batch_normalization_19_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp:assignvariableop_28_batch_normalization_19_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp$assignvariableop_29_conv2d_19_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_conv2d_19_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp#assignvariableop_31_dense_12_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp!assignvariableop_32_dense_12_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp#assignvariableop_33_dense_13_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp!assignvariableop_34_dense_13_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_14_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_14_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_16_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_16_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_16_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_16_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_17_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_17_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_17_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_17_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_18_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_18_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_18_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_18_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_19_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_19_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_19_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_19_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_12_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_12_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_13_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_13_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_14_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_14_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_16_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_16_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_16_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_16_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_17_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_17_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_17_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_17_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_18_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_18_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_18_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_18_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_19_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_19_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_19_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_19_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_12_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_12_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_13_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_13_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_829
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_83Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_83�
Identity_84IdentityIdentity_83:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_84"#
identity_84Identity_84:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_82AssignVariableOp_822(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������KK<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�

h2ptjl
_output
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__
	�call"�	
_tf_keras_model�{"name": "CNN3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN3", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN3"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
��
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer_with_weights-6
layer-8
layer_with_weights-7
layer-9
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"��
_tf_keras_sequential��{"name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_4_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT52AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 49, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_4_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_4_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT52AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 26}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 34}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 38}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 41}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 42}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 43}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 46}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 47}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 48}]}}}
�

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 51}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
"iter

#beta_1

$beta_2
	%decay
&learning_ratem�m�'m�(m�+m�,m�-m�.m�1m�2m�3m�4m�7m�8m�9m�:m�=m�>m�?m�@m�Am�Bm�v�v�'v�(v�+v�,v�-v�.v�1v�2v�3v�4v�7v�8v�9v�:v�=v�>v�?v�@v�Av�Bv�"
	optimizer
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
@25
A26
B27
28
29"
trackable_list_wrapper
�
'0
(1
+2
,3
-4
.5
16
27
38
49
710
811
912
:13
=14
>15
?16
@17
A18
B19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Elayer_metrics
	variables
trainable_variables
Flayer_regularization_losses
Gmetrics
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "lambda_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT52AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
�

Laxis
	'gamma
(beta
)moving_mean
*moving_variance
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

+kernel
,bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 55}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

Uaxis
	-gamma
.beta
/moving_mean
0moving_variance
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 32]}}
�


1kernel
2bias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 57}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 32]}}
�
^regularization_losses
_	variables
`trainable_variables
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 58}}
�

baxis
	3gamma
4beta
5moving_mean
6moving_variance
cregularization_losses
d	variables
etrainable_variables
f	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 128]}}
�


7kernel
8bias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 128]}}
�

kaxis
	9gamma
:beta
;moving_mean
<moving_variance
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}, "shared_object_id": 61}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 256]}}
�


=kernel
>bias
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 256]}}
�
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 63}}
�	

?kernel
@bias
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 41}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 700928}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 700928]}}
�
|regularization_losses
}	variables
~trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 43}
�	

Akernel
Bbias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 46}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 48}
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
@25
A26
B27"
trackable_list_wrapper
�
'0
(1
+2
,3
-4
.5
16
27
38
49
710
811
912
:13
=14
>15
?16
@17
A18
B19"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�layer_metrics
	variables
trainable_variables
 �layer_regularization_losses
�metrics
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_14/kernel
:2dense_14/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
regularization_losses
	variables
 trainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2batch_normalization_16/gamma
):'2batch_normalization_16/beta
2:0 (2"batch_normalization_16/moving_mean
6:4 (2&batch_normalization_16/moving_variance
*:( 2conv2d_16/kernel
: 2conv2d_16/bias
*:( 2batch_normalization_17/gamma
):' 2batch_normalization_17/beta
2:0  (2"batch_normalization_17/moving_mean
6:4  (2&batch_normalization_17/moving_variance
+:) �2conv2d_17/kernel
:�2conv2d_17/bias
+:)�2batch_normalization_18/gamma
*:(�2batch_normalization_18/beta
3:1� (2"batch_normalization_18/moving_mean
7:5� (2&batch_normalization_18/moving_variance
,:*��2conv2d_18/kernel
:�2conv2d_18/bias
+:)�2batch_normalization_19/gamma
*:(�2batch_normalization_19/beta
3:1� (2"batch_normalization_19/moving_mean
7:5� (2&batch_normalization_19/moving_variance
,:*��2conv2d_19/kernel
:�2conv2d_19/bias
$:"��*�2dense_12/kernel
:�2dense_12/bias
#:!
��2dense_13/kernel
:�2dense_13/bias
X
)0
*1
/2
03
54
65
;6
<7"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
�
�non_trainable_variables
�layer_metrics
Hregularization_losses
I	variables
Jtrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
Mregularization_losses
N	variables
Otrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
Qregularization_losses
R	variables
Strainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
Vregularization_losses
W	variables
Xtrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
Zregularization_losses
[	variables
\trainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
^regularization_losses
_	variables
`trainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
cregularization_losses
d	variables
etrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
gregularization_losses
h	variables
itrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
lregularization_losses
m	variables
ntrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
pregularization_losses
q	variables
rtrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
tregularization_losses
u	variables
vtrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
xregularization_losses
y	variables
ztrainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
|regularization_losses
}	variables
~trainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�regularization_losses
�	variables
�trainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�regularization_losses
�	variables
�trainable_variables
 �layer_regularization_losses
�metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
X
)0
*1
/2
03
54
65
;6
<7"
trackable_list_wrapper
�
	0

1
2
3
4
5
6
7
8
9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 66}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
':%	�2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
/:-2#Adam/batch_normalization_16/gamma/m
.:,2"Adam/batch_normalization_16/beta/m
/:- 2Adam/conv2d_16/kernel/m
!: 2Adam/conv2d_16/bias/m
/:- 2#Adam/batch_normalization_17/gamma/m
.:, 2"Adam/batch_normalization_17/beta/m
0:. �2Adam/conv2d_17/kernel/m
": �2Adam/conv2d_17/bias/m
0:.�2#Adam/batch_normalization_18/gamma/m
/:-�2"Adam/batch_normalization_18/beta/m
1:/��2Adam/conv2d_18/kernel/m
": �2Adam/conv2d_18/bias/m
0:.�2#Adam/batch_normalization_19/gamma/m
/:-�2"Adam/batch_normalization_19/beta/m
1:/��2Adam/conv2d_19/kernel/m
": �2Adam/conv2d_19/bias/m
):'��*�2Adam/dense_12/kernel/m
!:�2Adam/dense_12/bias/m
(:&
��2Adam/dense_13/kernel/m
!:�2Adam/dense_13/bias/m
':%	�2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
/:-2#Adam/batch_normalization_16/gamma/v
.:,2"Adam/batch_normalization_16/beta/v
/:- 2Adam/conv2d_16/kernel/v
!: 2Adam/conv2d_16/bias/v
/:- 2#Adam/batch_normalization_17/gamma/v
.:, 2"Adam/batch_normalization_17/beta/v
0:. �2Adam/conv2d_17/kernel/v
": �2Adam/conv2d_17/bias/v
0:.�2#Adam/batch_normalization_18/gamma/v
/:-�2"Adam/batch_normalization_18/beta/v
1:/��2Adam/conv2d_18/kernel/v
": �2Adam/conv2d_18/bias/v
0:.�2#Adam/batch_normalization_19/gamma/v
/:-�2"Adam/batch_normalization_19/beta/v
1:/��2Adam/conv2d_19/kernel/v
": �2Adam/conv2d_19/bias/v
):'��*�2Adam/dense_12/kernel/v
!:�2Adam/dense_12/bias/v
(:&
��2Adam/dense_13/kernel/v
!:�2Adam/dense_13/bias/v
�2�
A__inference_CNN3_layer_call_and_return_conditional_losses_1764411
A__inference_CNN3_layer_call_and_return_conditional_losses_1764561
A__inference_CNN3_layer_call_and_return_conditional_losses_1764697
A__inference_CNN3_layer_call_and_return_conditional_losses_1764847�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_1761933�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������KK
�2�
&__inference_CNN3_layer_call_fn_1764912
&__inference_CNN3_layer_call_fn_1764977
&__inference_CNN3_layer_call_fn_1765042
&__inference_CNN3_layer_call_fn_1765107�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference_call_1535237
__inference_call_1535355
__inference_call_1535473�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765254
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765397
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765526
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765669�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_sequential_4_layer_call_fn_1765730
.__inference_sequential_4_layer_call_fn_1765791
.__inference_sequential_4_layer_call_fn_1765852
.__inference_sequential_4_layer_call_fn_1765913�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_dense_14_layer_call_and_return_conditional_losses_1765924�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
*__inference_dense_14_layer_call_fn_1765933�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
%__inference_signature_wrapper_1764275input_1"�
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
�2�
E__inference_lambda_4_layer_call_and_return_conditional_losses_1765941
E__inference_lambda_4_layer_call_and_return_conditional_losses_1765949�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_lambda_4_layer_call_fn_1765954
*__inference_lambda_4_layer_call_fn_1765959�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1765977
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1765995
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1766013
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1766031�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_16_layer_call_fn_1766044
8__inference_batch_normalization_16_layer_call_fn_1766057
8__inference_batch_normalization_16_layer_call_fn_1766070
8__inference_batch_normalization_16_layer_call_fn_1766083�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1766106�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
+__inference_conv2d_16_layer_call_fn_1766115�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766133
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766151
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766169
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766187�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_17_layer_call_fn_1766200
8__inference_batch_normalization_17_layer_call_fn_1766213
8__inference_batch_normalization_17_layer_call_fn_1766226
8__inference_batch_normalization_17_layer_call_fn_1766239�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1766250�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
+__inference_conv2d_17_layer_call_fn_1766259�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1762191�
���
FullArgSpec
args�
jself
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
annotations� *@�=
;�84������������������������������������
�2�
1__inference_max_pooling2d_4_layer_call_fn_1762197�
���
FullArgSpec
args�
jself
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
annotations� *@�=
;�84������������������������������������
�2�
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766277
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766295
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766313
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766331�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_18_layer_call_fn_1766344
8__inference_batch_normalization_18_layer_call_fn_1766357
8__inference_batch_normalization_18_layer_call_fn_1766370
8__inference_batch_normalization_18_layer_call_fn_1766383�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_conv2d_18_layer_call_and_return_conditional_losses_1766394�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
+__inference_conv2d_18_layer_call_fn_1766403�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766421
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766439
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766457
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766475�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_19_layer_call_fn_1766488
8__inference_batch_normalization_19_layer_call_fn_1766501
8__inference_batch_normalization_19_layer_call_fn_1766514
8__inference_batch_normalization_19_layer_call_fn_1766527�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_conv2d_19_layer_call_and_return_conditional_losses_1766538�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
+__inference_conv2d_19_layer_call_fn_1766547�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
F__inference_flatten_4_layer_call_and_return_conditional_losses_1766553�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
+__inference_flatten_4_layer_call_fn_1766558�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
E__inference_dense_12_layer_call_and_return_conditional_losses_1766581�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
*__inference_dense_12_layer_call_fn_1766590�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
F__inference_dropout_8_layer_call_and_return_conditional_losses_1766595
F__inference_dropout_8_layer_call_and_return_conditional_losses_1766607�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dropout_8_layer_call_fn_1766612
+__inference_dropout_8_layer_call_fn_1766617�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_dense_13_layer_call_and_return_conditional_losses_1766640�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
*__inference_dense_13_layer_call_fn_1766649�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
F__inference_dropout_9_layer_call_and_return_conditional_losses_1766654
F__inference_dropout_9_layer_call_and_return_conditional_losses_1766666�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dropout_9_layer_call_fn_1766671
+__inference_dropout_9_layer_call_fn_1766676�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference_loss_fn_0_1766687�
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
�2�
__inference_loss_fn_1_1766698�
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
�2�
__inference_loss_fn_2_1766709�
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
annotations� *� �
A__inference_CNN3_layer_call_and_return_conditional_losses_1764411�'()*+,-./0123456789:;<=>?@AB;�8
1�.
(�%
inputs���������KK
p 
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_1764561�'()*+,-./0123456789:;<=>?@AB;�8
1�.
(�%
inputs���������KK
p
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_1764697�'()*+,-./0123456789:;<=>?@AB<�9
2�/
)�&
input_1���������KK
p 
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_1764847�'()*+,-./0123456789:;<=>?@AB<�9
2�/
)�&
input_1���������KK
p
� "%�"
�
0���������
� �
&__inference_CNN3_layer_call_fn_1764912x'()*+,-./0123456789:;<=>?@AB<�9
2�/
)�&
input_1���������KK
p 
� "�����������
&__inference_CNN3_layer_call_fn_1764977w'()*+,-./0123456789:;<=>?@AB;�8
1�.
(�%
inputs���������KK
p 
� "�����������
&__inference_CNN3_layer_call_fn_1765042w'()*+,-./0123456789:;<=>?@AB;�8
1�.
(�%
inputs���������KK
p
� "�����������
&__inference_CNN3_layer_call_fn_1765107x'()*+,-./0123456789:;<=>?@AB<�9
2�/
)�&
input_1���������KK
p
� "�����������
"__inference__wrapped_model_1761933�'()*+,-./0123456789:;<=>?@AB8�5
.�+
)�&
input_1���������KK
� "3�0
.
output_1"�
output_1����������
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1765977�'()*M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1765995�'()*M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1766013r'()*;�8
1�.
(�%
inputs���������KK
p 
� "-�*
#� 
0���������KK
� �
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1766031r'()*;�8
1�.
(�%
inputs���������KK
p
� "-�*
#� 
0���������KK
� �
8__inference_batch_normalization_16_layer_call_fn_1766044�'()*M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
8__inference_batch_normalization_16_layer_call_fn_1766057�'()*M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
8__inference_batch_normalization_16_layer_call_fn_1766070e'()*;�8
1�.
(�%
inputs���������KK
p 
� " ����������KK�
8__inference_batch_normalization_16_layer_call_fn_1766083e'()*;�8
1�.
(�%
inputs���������KK
p
� " ����������KK�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766133�-./0M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766151�-./0M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766169r-./0;�8
1�.
(�%
inputs���������KK 
p 
� "-�*
#� 
0���������KK 
� �
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1766187r-./0;�8
1�.
(�%
inputs���������KK 
p
� "-�*
#� 
0���������KK 
� �
8__inference_batch_normalization_17_layer_call_fn_1766200�-./0M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
8__inference_batch_normalization_17_layer_call_fn_1766213�-./0M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
8__inference_batch_normalization_17_layer_call_fn_1766226e-./0;�8
1�.
(�%
inputs���������KK 
p 
� " ����������KK �
8__inference_batch_normalization_17_layer_call_fn_1766239e-./0;�8
1�.
(�%
inputs���������KK 
p
� " ����������KK �
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766277�3456N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766295�3456N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766313t3456<�9
2�/
)�&
inputs���������%%�
p 
� ".�+
$�!
0���������%%�
� �
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1766331t3456<�9
2�/
)�&
inputs���������%%�
p
� ".�+
$�!
0���������%%�
� �
8__inference_batch_normalization_18_layer_call_fn_1766344�3456N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_18_layer_call_fn_1766357�3456N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_18_layer_call_fn_1766370g3456<�9
2�/
)�&
inputs���������%%�
p 
� "!����������%%��
8__inference_batch_normalization_18_layer_call_fn_1766383g3456<�9
2�/
)�&
inputs���������%%�
p
� "!����������%%��
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766421�9:;<N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766439�9:;<N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766457t9:;<<�9
2�/
)�&
inputs���������%%�
p 
� ".�+
$�!
0���������%%�
� �
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1766475t9:;<<�9
2�/
)�&
inputs���������%%�
p
� ".�+
$�!
0���������%%�
� �
8__inference_batch_normalization_19_layer_call_fn_1766488�9:;<N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_19_layer_call_fn_1766501�9:;<N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_19_layer_call_fn_1766514g9:;<<�9
2�/
)�&
inputs���������%%�
p 
� "!����������%%��
8__inference_batch_normalization_19_layer_call_fn_1766527g9:;<<�9
2�/
)�&
inputs���������%%�
p
� "!����������%%��
__inference_call_1535237g'()*+,-./0123456789:;<=>?@AB3�0
)�&
 �
inputs�KK
p
� "�	��
__inference_call_1535355g'()*+,-./0123456789:;<=>?@AB3�0
)�&
 �
inputs�KK
p 
� "�	��
__inference_call_1535473w'()*+,-./0123456789:;<=>?@AB;�8
1�.
(�%
inputs���������KK
p 
� "�����������
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1766106l+,7�4
-�*
(�%
inputs���������KK
� "-�*
#� 
0���������KK 
� �
+__inference_conv2d_16_layer_call_fn_1766115_+,7�4
-�*
(�%
inputs���������KK
� " ����������KK �
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1766250m127�4
-�*
(�%
inputs���������KK 
� ".�+
$�!
0���������KK�
� �
+__inference_conv2d_17_layer_call_fn_1766259`127�4
-�*
(�%
inputs���������KK 
� "!����������KK��
F__inference_conv2d_18_layer_call_and_return_conditional_losses_1766394n788�5
.�+
)�&
inputs���������%%�
� ".�+
$�!
0���������%%�
� �
+__inference_conv2d_18_layer_call_fn_1766403a788�5
.�+
)�&
inputs���������%%�
� "!����������%%��
F__inference_conv2d_19_layer_call_and_return_conditional_losses_1766538n=>8�5
.�+
)�&
inputs���������%%�
� ".�+
$�!
0���������%%�
� �
+__inference_conv2d_19_layer_call_fn_1766547a=>8�5
.�+
)�&
inputs���������%%�
� "!����������%%��
E__inference_dense_12_layer_call_and_return_conditional_losses_1766581_?@1�.
'�$
"�
inputs�����������*
� "&�#
�
0����������
� �
*__inference_dense_12_layer_call_fn_1766590R?@1�.
'�$
"�
inputs�����������*
� "������������
E__inference_dense_13_layer_call_and_return_conditional_losses_1766640^AB0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_13_layer_call_fn_1766649QAB0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_14_layer_call_and_return_conditional_losses_1765924]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_14_layer_call_fn_1765933P0�-
&�#
!�
inputs����������
� "�����������
F__inference_dropout_8_layer_call_and_return_conditional_losses_1766595^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_dropout_8_layer_call_and_return_conditional_losses_1766607^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
+__inference_dropout_8_layer_call_fn_1766612Q4�1
*�'
!�
inputs����������
p 
� "������������
+__inference_dropout_8_layer_call_fn_1766617Q4�1
*�'
!�
inputs����������
p
� "������������
F__inference_dropout_9_layer_call_and_return_conditional_losses_1766654^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_dropout_9_layer_call_and_return_conditional_losses_1766666^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
+__inference_dropout_9_layer_call_fn_1766671Q4�1
*�'
!�
inputs����������
p 
� "������������
+__inference_dropout_9_layer_call_fn_1766676Q4�1
*�'
!�
inputs����������
p
� "������������
F__inference_flatten_4_layer_call_and_return_conditional_losses_1766553c8�5
.�+
)�&
inputs���������%%�
� "'�$
�
0�����������*
� �
+__inference_flatten_4_layer_call_fn_1766558V8�5
.�+
)�&
inputs���������%%�
� "������������*�
E__inference_lambda_4_layer_call_and_return_conditional_losses_1765941p?�<
5�2
(�%
inputs���������KK

 
p 
� "-�*
#� 
0���������KK
� �
E__inference_lambda_4_layer_call_and_return_conditional_losses_1765949p?�<
5�2
(�%
inputs���������KK

 
p
� "-�*
#� 
0���������KK
� �
*__inference_lambda_4_layer_call_fn_1765954c?�<
5�2
(�%
inputs���������KK

 
p 
� " ����������KK�
*__inference_lambda_4_layer_call_fn_1765959c?�<
5�2
(�%
inputs���������KK

 
p
� " ����������KK<
__inference_loss_fn_0_1766687+�

� 
� "� <
__inference_loss_fn_1_1766698?�

� 
� "� <
__inference_loss_fn_2_1766709A�

� 
� "� �
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1762191�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_4_layer_call_fn_1762197�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765254�'()*+,-./0123456789:;<=>?@AB?�<
5�2
(�%
inputs���������KK
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765397�'()*+,-./0123456789:;<=>?@AB?�<
5�2
(�%
inputs���������KK
p

 
� "&�#
�
0����������
� �
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765526�'()*+,-./0123456789:;<=>?@ABG�D
=�:
0�-
lambda_4_input���������KK
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_4_layer_call_and_return_conditional_losses_1765669�'()*+,-./0123456789:;<=>?@ABG�D
=�:
0�-
lambda_4_input���������KK
p

 
� "&�#
�
0����������
� �
.__inference_sequential_4_layer_call_fn_1765730�'()*+,-./0123456789:;<=>?@ABG�D
=�:
0�-
lambda_4_input���������KK
p 

 
� "������������
.__inference_sequential_4_layer_call_fn_1765791z'()*+,-./0123456789:;<=>?@AB?�<
5�2
(�%
inputs���������KK
p 

 
� "������������
.__inference_sequential_4_layer_call_fn_1765852z'()*+,-./0123456789:;<=>?@AB?�<
5�2
(�%
inputs���������KK
p

 
� "������������
.__inference_sequential_4_layer_call_fn_1765913�'()*+,-./0123456789:;<=>?@ABG�D
=�:
0�-
lambda_4_input���������KK
p

 
� "������������
%__inference_signature_wrapper_1764275�'()*+,-./0123456789:;<=>?@ABC�@
� 
9�6
4
input_1)�&
input_1���������KK"3�0
.
output_1"�
output_1���������