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
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	�*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
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
batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_24/gamma
�
0batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_24/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_24/beta
�
/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_24/beta*
_output_shapes
:*
dtype0
�
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
: *
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
: *
dtype0
�
batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_25/gamma
�
0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_25/beta
�
/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes
: *
dtype0
�
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*!
shared_nameconv2d_25/kernel
~
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*'
_output_shapes
: �*
dtype0
u
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_25/bias
n
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_26/gamma
�
0batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_26/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_26/beta
�
/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_26/beta*
_output_shapes	
:�*
dtype0
�
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_26/kernel

$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_26/bias
n
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_27/gamma
�
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_27/beta
�
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes	
:�*
dtype0
�
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_27/kernel

$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_27/bias
n
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes	
:�*
dtype0
}
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*�* 
shared_namedense_18/kernel
v
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*!
_output_shapes
:��*�*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:�*
dtype0
|
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_19/kernel
u
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel* 
_output_shapes
:
��*
dtype0
s
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_19/bias
l
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes	
:�*
dtype0
�
"batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_24/moving_mean
�
6batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_24/moving_mean*
_output_shapes
:*
dtype0
�
&batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_24/moving_variance
�
:batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_24/moving_variance*
_output_shapes
:*
dtype0
�
"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_25/moving_mean
�
6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_25/moving_variance
�
:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes
: *
dtype0
�
"batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_26/moving_mean
�
6batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_26/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_26/moving_variance
�
:batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_26/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_27/moving_mean
�
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_27/moving_variance
�
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes	
:�*
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
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_20/kernel/m
�
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_24/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_24/gamma/m
�
7Adam/batch_normalization_24/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_24/gamma/m*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_24/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_24/beta/m
�
6Adam/batch_normalization_24/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_24/beta/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_24/kernel/m
�
+Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_24/bias/m
{
)Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_25/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_25/gamma/m
�
7Adam/batch_normalization_25/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_25/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_25/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_25/beta/m
�
6Adam/batch_normalization_25/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_25/beta/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_25/kernel/m
�
+Adam/conv2d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/m*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_25/bias/m
|
)Adam/conv2d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_26/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_26/gamma/m
�
7Adam/batch_normalization_26/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_26/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_26/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_26/beta/m
�
6Adam/batch_normalization_26/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_26/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_26/kernel/m
�
+Adam/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_26/bias/m
|
)Adam/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_27/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_27/gamma/m
�
7Adam/batch_normalization_27/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_27/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_27/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_27/beta/m
�
6Adam/batch_normalization_27/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_27/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_27/kernel/m
�
+Adam/conv2d_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_27/bias/m
|
)Adam/conv2d_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*�*'
shared_nameAdam/dense_18/kernel/m
�
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*!
_output_shapes
:��*�*
dtype0
�
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_18/bias/m
z
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_19/kernel/m
�
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_19/bias/m
z
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_20/kernel/v
�
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_24/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_24/gamma/v
�
7Adam/batch_normalization_24/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_24/gamma/v*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_24/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_24/beta/v
�
6Adam/batch_normalization_24/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_24/beta/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_24/kernel/v
�
+Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_24/bias/v
{
)Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_25/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_25/gamma/v
�
7Adam/batch_normalization_25/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_25/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_25/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_25/beta/v
�
6Adam/batch_normalization_25/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_25/beta/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_25/kernel/v
�
+Adam/conv2d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/v*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_25/bias/v
|
)Adam/conv2d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_26/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_26/gamma/v
�
7Adam/batch_normalization_26/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_26/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_26/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_26/beta/v
�
6Adam/batch_normalization_26/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_26/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_26/kernel/v
�
+Adam/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_26/bias/v
|
)Adam/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_27/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_27/gamma/v
�
7Adam/batch_normalization_27/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_27/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_27/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_27/beta/v
�
6Adam/batch_normalization_27/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_27/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_27/kernel/v
�
+Adam/conv2d_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_27/bias/v
|
)Adam/conv2d_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*�*'
shared_nameAdam/dense_18/kernel/v
�
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*!
_output_shapes
:��*�*
dtype0
�
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_18/bias/v
z
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_19/kernel/v
�
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_19/bias/v
z
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
ԇ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�

h2ptjl
_output
	optimizer
trainable_variables
regularization_losses
	variables
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
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
�
"iter

#beta_1

$beta_2
	%decay
&learning_ratem�m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�4m�5m�6m�7m�8m�9m�:m�v�v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�4v�5v�6v�7v�8v�9v�:v�
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
20
21
 
�
'0
(1
;2
<3
)4
*5
+6
,7
=8
>9
-10
.11
/12
013
?14
@15
116
217
318
419
A20
B21
522
623
724
825
926
:27
28
29
�
trainable_variables

Clayers
Dlayer_metrics
Elayer_regularization_losses
regularization_losses
Fmetrics
	variables
Gnon_trainable_variables
 
R
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
�
Laxis
	'gamma
(beta
;moving_mean
<moving_variance
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
h

)kernel
*bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
�
Uaxis
	+gamma
,beta
=moving_mean
>moving_variance
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
h

-kernel
.bias
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
R
^trainable_variables
_regularization_losses
`	variables
a	keras_api
�
baxis
	/gamma
0beta
?moving_mean
@moving_variance
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
h

1kernel
2bias
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
�
kaxis
	3gamma
4beta
Amoving_mean
Bmoving_variance
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
h

5kernel
6bias
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
R
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
h

7kernel
8bias
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
R
|trainable_variables
}regularization_losses
~	variables
	keras_api
l

9kernel
:bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
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
 
�
'0
(1
;2
<3
)4
*5
+6
,7
=8
>9
-10
.11
/12
013
?14
@15
116
217
318
419
A20
B21
522
623
724
825
926
:27
�
trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
regularization_losses
�metrics
	variables
�non_trainable_variables
NL
VARIABLE_VALUEdense_20/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_20/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
�layers
�layer_metrics
regularization_losses
�metrics
�non_trainable_variables
 	variables
 �layer_regularization_losses
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
b`
VARIABLE_VALUEbatch_normalization_24/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_24/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_24/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_24/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_25/gamma0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_25/beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_25/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_25/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_26/gamma0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_26/beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_26/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_26/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_27/gamma1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_27/beta1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_27/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_27/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_18/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_18/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_19/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_19/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_24/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_24/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_25/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_25/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_26/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_26/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_27/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_27/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 

�0
�1
8
;0
<1
=2
>3
?4
@5
A6
B7
 
 
 
�
Htrainable_variables
�layers
�layer_metrics
Iregularization_losses
�metrics
�non_trainable_variables
J	variables
 �layer_regularization_losses
 

'0
(1
 

'0
(1
;2
<3
�
Mtrainable_variables
�layers
�layer_metrics
Nregularization_losses
�metrics
�non_trainable_variables
O	variables
 �layer_regularization_losses

)0
*1
 

)0
*1
�
Qtrainable_variables
�layers
�layer_metrics
Rregularization_losses
�metrics
�non_trainable_variables
S	variables
 �layer_regularization_losses
 

+0
,1
 

+0
,1
=2
>3
�
Vtrainable_variables
�layers
�layer_metrics
Wregularization_losses
�metrics
�non_trainable_variables
X	variables
 �layer_regularization_losses

-0
.1
 

-0
.1
�
Ztrainable_variables
�layers
�layer_metrics
[regularization_losses
�metrics
�non_trainable_variables
\	variables
 �layer_regularization_losses
 
 
 
�
^trainable_variables
�layers
�layer_metrics
_regularization_losses
�metrics
�non_trainable_variables
`	variables
 �layer_regularization_losses
 

/0
01
 

/0
01
?2
@3
�
ctrainable_variables
�layers
�layer_metrics
dregularization_losses
�metrics
�non_trainable_variables
e	variables
 �layer_regularization_losses

10
21
 

10
21
�
gtrainable_variables
�layers
�layer_metrics
hregularization_losses
�metrics
�non_trainable_variables
i	variables
 �layer_regularization_losses
 

30
41
 

30
41
A2
B3
�
ltrainable_variables
�layers
�layer_metrics
mregularization_losses
�metrics
�non_trainable_variables
n	variables
 �layer_regularization_losses

50
61
 

50
61
�
ptrainable_variables
�layers
�layer_metrics
qregularization_losses
�metrics
�non_trainable_variables
r	variables
 �layer_regularization_losses
 
 
 
�
ttrainable_variables
�layers
�layer_metrics
uregularization_losses
�metrics
�non_trainable_variables
v	variables
 �layer_regularization_losses

70
81
 

70
81
�
xtrainable_variables
�layers
�layer_metrics
yregularization_losses
�metrics
�non_trainable_variables
z	variables
 �layer_regularization_losses
 
 
 
�
|trainable_variables
�layers
�layer_metrics
}regularization_losses
�metrics
�non_trainable_variables
~	variables
 �layer_regularization_losses

90
:1
 

90
:1
�
�trainable_variables
�layers
�layer_metrics
�regularization_losses
�metrics
�non_trainable_variables
�	variables
 �layer_regularization_losses
 
 
 
�
�trainable_variables
�layers
�layer_metrics
�regularization_losses
�metrics
�non_trainable_variables
�	variables
 �layer_regularization_losses
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
8
;0
<1
=2
>3
?4
@5
A6
B7
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

=0
>1
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
?0
@1
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
A0
B1
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
VARIABLE_VALUEAdam/dense_20/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_20/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_24/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_24/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_24/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_24/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_25/gamma/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_25/beta/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_25/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_25/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_26/gamma/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_26/beta/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_26/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_26/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_27/gamma/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_27/beta/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_27/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_27/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_18/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_18/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_19/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_19/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_20/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_20/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_24/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_24/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_24/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_24/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_25/gamma/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_25/beta/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_25/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_25/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_26/gamma/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_26/beta/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_26/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_26/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_27/gamma/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_27/beta/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_27/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_27/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_18/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_18/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_19/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_19/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������KK*
dtype0*$
shape:���������KK
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_24/kernelconv2d_24/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_25/kernelconv2d_25/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_26/kernelconv2d_26/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_27/kernelconv2d_27/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias**
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
%__inference_signature_wrapper_3196763
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
� 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0batch_normalization_24/gamma/Read/ReadVariableOp/batch_normalization_24/beta/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp0batch_normalization_26/gamma/Read/ReadVariableOp/batch_normalization_26/beta/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp6batch_normalization_24/moving_mean/Read/ReadVariableOp:batch_normalization_24/moving_variance/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOp6batch_normalization_26/moving_mean/Read/ReadVariableOp:batch_normalization_26/moving_variance/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp7Adam/batch_normalization_24/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_24/beta/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp7Adam/batch_normalization_25/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_25/beta/m/Read/ReadVariableOp+Adam/conv2d_25/kernel/m/Read/ReadVariableOp)Adam/conv2d_25/bias/m/Read/ReadVariableOp7Adam/batch_normalization_26/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_26/beta/m/Read/ReadVariableOp+Adam/conv2d_26/kernel/m/Read/ReadVariableOp)Adam/conv2d_26/bias/m/Read/ReadVariableOp7Adam/batch_normalization_27/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_27/beta/m/Read/ReadVariableOp+Adam/conv2d_27/kernel/m/Read/ReadVariableOp)Adam/conv2d_27/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp7Adam/batch_normalization_24/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_24/beta/v/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp7Adam/batch_normalization_25/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_25/beta/v/Read/ReadVariableOp+Adam/conv2d_25/kernel/v/Read/ReadVariableOp)Adam/conv2d_25/bias/v/Read/ReadVariableOp7Adam/batch_normalization_26/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_26/beta/v/Read/ReadVariableOp+Adam/conv2d_26/kernel/v/Read/ReadVariableOp)Adam/conv2d_26/bias/v/Read/ReadVariableOp7Adam/batch_normalization_27/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_27/beta/v/Read/ReadVariableOp+Adam/conv2d_27/kernel/v/Read/ReadVariableOp)Adam/conv2d_27/bias/v/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOpConst*`
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
 __inference__traced_save_3199469
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_24/gammabatch_normalization_24/betaconv2d_24/kernelconv2d_24/biasbatch_normalization_25/gammabatch_normalization_25/betaconv2d_25/kernelconv2d_25/biasbatch_normalization_26/gammabatch_normalization_26/betaconv2d_26/kernelconv2d_26/biasbatch_normalization_27/gammabatch_normalization_27/betaconv2d_27/kernelconv2d_27/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias"batch_normalization_24/moving_mean&batch_normalization_24/moving_variance"batch_normalization_25/moving_mean&batch_normalization_25/moving_variance"batch_normalization_26/moving_mean&batch_normalization_26/moving_variance"batch_normalization_27/moving_mean&batch_normalization_27/moving_variancetotalcounttotal_1count_1Adam/dense_20/kernel/mAdam/dense_20/bias/m#Adam/batch_normalization_24/gamma/m"Adam/batch_normalization_24/beta/mAdam/conv2d_24/kernel/mAdam/conv2d_24/bias/m#Adam/batch_normalization_25/gamma/m"Adam/batch_normalization_25/beta/mAdam/conv2d_25/kernel/mAdam/conv2d_25/bias/m#Adam/batch_normalization_26/gamma/m"Adam/batch_normalization_26/beta/mAdam/conv2d_26/kernel/mAdam/conv2d_26/bias/m#Adam/batch_normalization_27/gamma/m"Adam/batch_normalization_27/beta/mAdam/conv2d_27/kernel/mAdam/conv2d_27/bias/mAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/mAdam/dense_20/kernel/vAdam/dense_20/bias/v#Adam/batch_normalization_24/gamma/v"Adam/batch_normalization_24/beta/vAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/v#Adam/batch_normalization_25/gamma/v"Adam/batch_normalization_25/beta/vAdam/conv2d_25/kernel/vAdam/conv2d_25/bias/v#Adam/batch_normalization_26/gamma/v"Adam/batch_normalization_26/beta/vAdam/conv2d_26/kernel/vAdam/conv2d_26/bias/v#Adam/batch_normalization_27/gamma/v"Adam/batch_normalization_27/beta/vAdam/conv2d_27/kernel/vAdam/conv2d_27/bias/vAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/v*_
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
#__inference__traced_restore_3199728��%
�
�
8__inference_batch_normalization_24_layer_call_fn_3198558

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_31949712
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
�
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198963

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
�
e
G__inference_dropout_13_layer_call_and_return_conditional_losses_3199142

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
f
G__inference_dropout_13_layer_call_and_return_conditional_losses_3199154

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
�
�
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3194971

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
8__inference_batch_normalization_24_layer_call_fn_3198532

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_31944432
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
8__inference_batch_normalization_24_layer_call_fn_3198545

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_31944872
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
�
e
,__inference_dropout_12_layer_call_fn_3199105

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
GPU2*0J 8� *P
fKRI
G__inference_dropout_12_layer_call_and_return_conditional_losses_31953362
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
��
�
__inference_call_2939064

inputsI
;sequential_6_batch_normalization_24_readvariableop_resource:K
=sequential_6_batch_normalization_24_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_25_readvariableop_resource: K
=sequential_6_batch_normalization_25_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_6_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_25_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_26_readvariableop_resource:	�L
=sequential_6_batch_normalization_26_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_26_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_27_readvariableop_resource:	�L
=sequential_6_batch_normalization_27_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_27_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_27_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:��*�D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_24/ReadVariableOp�4sequential_6/batch_normalization_24/ReadVariableOp_1�Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_25/ReadVariableOp�4sequential_6/batch_normalization_25/ReadVariableOp_1�Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_26/ReadVariableOp�4sequential_6/batch_normalization_26/ReadVariableOp_1�Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_27/ReadVariableOp�4sequential_6/batch_normalization_27/ReadVariableOp_1�-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�,sequential_6/conv2d_24/Conv2D/ReadVariableOp�-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�,sequential_6/conv2d_25/Conv2D/ReadVariableOp�-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�,sequential_6/conv2d_26/Conv2D/ReadVariableOp�-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�,sequential_6/conv2d_27/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
)sequential_6/lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_6/lambda_6/strided_slice/stack�
+sequential_6/lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_6/lambda_6/strided_slice/stack_1�
+sequential_6/lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_6/lambda_6/strided_slice/stack_2�
#sequential_6/lambda_6/strided_sliceStridedSliceinputs2sequential_6/lambda_6/strided_slice/stack:output:04sequential_6/lambda_6/strided_slice/stack_1:output:04sequential_6/lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_6/lambda_6/strided_slice�
2sequential_6/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_6/batch_normalization_24/ReadVariableOp�
4sequential_6/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_6/batch_normalization_24/ReadVariableOp_1�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:0:sequential_6/batch_normalization_24/ReadVariableOp:value:0<sequential_6/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_24/FusedBatchNormV3�
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp�
sequential_6/conv2d_24/Conv2DConv2D8sequential_6/batch_normalization_24/FusedBatchNormV3:y:04sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D�
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_24/BiasAdd�
sequential_6/conv2d_24/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_24/Relu�
2sequential_6/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_6/batch_normalization_25/ReadVariableOp�
4sequential_6/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_6/batch_normalization_25/ReadVariableOp_1�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_24/Relu:activations:0:sequential_6/batch_normalization_25/ReadVariableOp:value:0<sequential_6/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_25/FusedBatchNormV3�
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp�
sequential_6/conv2d_25/Conv2DConv2D8sequential_6/batch_normalization_25/FusedBatchNormV3:y:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_6/conv2d_25/Conv2D�
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_6/conv2d_25/BiasAdd�
sequential_6/conv2d_25/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_6/conv2d_25/Relu�
$sequential_6/max_pooling2d_6/MaxPoolMaxPool)sequential_6/conv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_6/max_pooling2d_6/MaxPool�
2sequential_6/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_26/ReadVariableOp�
4sequential_6/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_26/ReadVariableOp_1�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3-sequential_6/max_pooling2d_6/MaxPool:output:0:sequential_6/batch_normalization_26/ReadVariableOp:value:0<sequential_6/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_26/FusedBatchNormV3�
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp�
sequential_6/conv2d_26/Conv2DConv2D8sequential_6/batch_normalization_26/FusedBatchNormV3:y:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D�
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_26/BiasAdd�
sequential_6/conv2d_26/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_26/Relu�
2sequential_6/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_27/ReadVariableOp�
4sequential_6/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_27/ReadVariableOp_1�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_26/Relu:activations:0:sequential_6/batch_normalization_27/ReadVariableOp:value:0<sequential_6/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_27/FusedBatchNormV3�
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp�
sequential_6/conv2d_27/Conv2DConv2D8sequential_6/batch_normalization_27/FusedBatchNormV3:y:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_27/Conv2D�
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_27/BiasAdd�
sequential_6/conv2d_27/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_27/Relu�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/conv2d_27/Relu:activations:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOp�
sequential_6/dense_18/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/MatMul�
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOp�
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/BiasAdd�
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/Relu�
 sequential_6/dropout_12/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_12/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_12/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/MatMul�
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOp�
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/BiasAdd�
sequential_6/dense_19/ReluRelu&sequential_6/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/Relu�
 sequential_6/dropout_13/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_13/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_13/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdd|
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_20/Softmax�
IdentityIdentitydense_20/Softmax:softmax:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpD^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_24/ReadVariableOp5^sequential_6/batch_normalization_24/ReadVariableOp_1D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_25/ReadVariableOp5^sequential_6/batch_normalization_25/ReadVariableOp_1D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_26/ReadVariableOp5^sequential_6/batch_normalization_26/ReadVariableOp_1D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_27/ReadVariableOp5^sequential_6/batch_normalization_27/ReadVariableOp_1.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_24/ReadVariableOp2sequential_6/batch_normalization_24/ReadVariableOp2l
4sequential_6/batch_normalization_24/ReadVariableOp_14sequential_6/batch_normalization_24/ReadVariableOp_12�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_25/ReadVariableOp2sequential_6/batch_normalization_25/ReadVariableOp2l
4sequential_6/batch_normalization_25/ReadVariableOp_14sequential_6/batch_normalization_25/ReadVariableOp_12�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_26/ReadVariableOp2sequential_6/batch_normalization_26/ReadVariableOp2l
4sequential_6/batch_normalization_26/ReadVariableOp_14sequential_6/batch_normalization_26/ReadVariableOp_12�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_27/ReadVariableOp2sequential_6/batch_normalization_27/ReadVariableOp2l
4sequential_6/batch_normalization_27/ReadVariableOp_14sequential_6/batch_normalization_27/ReadVariableOp_12^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
E__inference_dense_18_layer_call_and_return_conditional_losses_3199069

inputs3
matmul_readvariableop_resource:��*�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������*
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_27_layer_call_fn_3199002

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_31951102
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
�
�
F__inference_conv2d_27_layer_call_and_return_conditional_losses_3195131

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
8__inference_batch_normalization_27_layer_call_fn_3198976

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_31948332
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
�
H
,__inference_dropout_13_layer_call_fn_3199159

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
GPU2*0J 8� *P
fKRI
G__inference_dropout_13_layer_call_and_return_conditional_losses_31952032
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
�
f
G__inference_dropout_12_layer_call_and_return_conditional_losses_3199095

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
�
�
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198783

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
�
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_3198437

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
�
�
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3194443

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
�
F
*__inference_lambda_6_layer_call_fn_3198442

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
E__inference_lambda_6_layer_call_and_return_conditional_losses_31949522
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
��
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_3198157
lambda_6_input<
.batch_normalization_24_readvariableop_resource:>
0batch_normalization_24_readvariableop_1_resource:M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: <
.batch_normalization_25_readvariableop_resource: >
0batch_normalization_25_readvariableop_1_resource: M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: C
(conv2d_25_conv2d_readvariableop_resource: �8
)conv2d_25_biasadd_readvariableop_resource:	�=
.batch_normalization_26_readvariableop_resource:	�?
0batch_normalization_26_readvariableop_1_resource:	�N
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�=
.batch_normalization_27_readvariableop_resource:	�?
0batch_normalization_27_readvariableop_1_resource:	�N
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_27_conv2d_readvariableop_resource:��8
)conv2d_27_biasadd_readvariableop_resource:	�<
'dense_18_matmul_readvariableop_resource:��*�7
(dense_18_biasadd_readvariableop_resource:	�;
'dense_19_matmul_readvariableop_resource:
��7
(dense_19_biasadd_readvariableop_resource:	�
identity��%batch_normalization_24/AssignNewValue�'batch_normalization_24/AssignNewValue_1�6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_24/ReadVariableOp�'batch_normalization_24/ReadVariableOp_1�%batch_normalization_25/AssignNewValue�'batch_normalization_25/AssignNewValue_1�6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_25/ReadVariableOp�'batch_normalization_25/ReadVariableOp_1�%batch_normalization_26/AssignNewValue�'batch_normalization_26/AssignNewValue_1�6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_26/ReadVariableOp�'batch_normalization_26/ReadVariableOp_1�%batch_normalization_27/AssignNewValue�'batch_normalization_27/AssignNewValue_1�6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_27/ReadVariableOp�'batch_normalization_27/ReadVariableOp_1� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_6/strided_slice/stack�
lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_6/strided_slice/stack_1�
lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_6/strided_slice/stack_2�
lambda_6/strided_sliceStridedSlicelambda_6_input%lambda_6/strided_slice/stack:output:0'lambda_6/strided_slice/stack_1:output:0'lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_6/strided_slice�
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_24/ReadVariableOp�
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_24/ReadVariableOp_1�
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3lambda_6/strided_slice:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_24/FusedBatchNormV3�
%batch_normalization_24/AssignNewValueAssignVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource4batch_normalization_24/FusedBatchNormV3:batch_mean:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_24/AssignNewValue�
'batch_normalization_24/AssignNewValue_1AssignVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_24/FusedBatchNormV3:batch_variance:09^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_24/AssignNewValue_1�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_24/Conv2D�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_24/Relu�
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_25/ReadVariableOp�
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_25/ReadVariableOp_1�
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_24/Relu:activations:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_25/FusedBatchNormV3�
%batch_normalization_25/AssignNewValueAssignVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource4batch_normalization_25/FusedBatchNormV3:batch_mean:07^batch_normalization_25/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_25/AssignNewValue�
'batch_normalization_25/AssignNewValue_1AssignVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_25/FusedBatchNormV3:batch_variance:09^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_25/AssignNewValue_1�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_25/Conv2D/ReadVariableOp�
conv2d_25/Conv2DConv2D+batch_normalization_25/FusedBatchNormV3:y:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
conv2d_25/Conv2D�
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp�
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
conv2d_25/Relu�
max_pooling2d_6/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool�
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_26/ReadVariableOp�
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_26/ReadVariableOp_1�
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_6/MaxPool:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_26/FusedBatchNormV3�
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_26/AssignNewValue�
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_26/AssignNewValue_1�
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_26/Conv2D/ReadVariableOp�
conv2d_26/Conv2DConv2D+batch_normalization_26/FusedBatchNormV3:y:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_26/Conv2D�
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp�
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_26/Relu�
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_27/ReadVariableOp�
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_27/ReadVariableOp_1�
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_26/Relu:activations:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_27/FusedBatchNormV3�
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue�
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1�
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2D+batch_normalization_27/FusedBatchNormV3:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_27/BiasAdd
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_27/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
flatten_6/Const�
flatten_6/ReshapeReshapeconv2d_27/Relu:activations:0flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2
flatten_6/Reshape�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulflatten_6/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_18/Reluy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_12/dropout/Const�
dropout_12/dropout/MulMuldense_18/Relu:activations:0!dropout_12/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_12/dropout/Mul
dropout_12/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shape�
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform�
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_12/dropout/GreaterEqual/y�
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_12/dropout/GreaterEqual�
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_12/dropout/Cast�
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_12/dropout/Mul_1�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_12/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/BiasAddt
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_19/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Const�
dropout_13/dropout/MulMuldense_19/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape�
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform�
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/y�
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_13/dropout/GreaterEqual�
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_13/dropout/Cast�
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_13/dropout/Mul_1�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentitydropout_13/dropout/Mul_1:z:0&^batch_normalization_24/AssignNewValue(^batch_normalization_24/AssignNewValue_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_1&^batch_normalization_25/AssignNewValue(^batch_normalization_25/AssignNewValue_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_24/AssignNewValue%batch_normalization_24/AssignNewValue2R
'batch_normalization_24/AssignNewValue_1'batch_normalization_24/AssignNewValue_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12N
%batch_normalization_25/AssignNewValue%batch_normalization_25/AssignNewValue2R
'batch_normalization_25/AssignNewValue_1'batch_normalization_25/AssignNewValue_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_6_input
�
�
.__inference_sequential_6_layer_call_fn_3198218
lambda_6_input
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
StatefulPartitionedCallStatefulPartitionedCalllambda_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_31952242
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
_user_specified_namelambda_6_input
�
�
.__inference_sequential_6_layer_call_fn_3198401
lambda_6_input
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
StatefulPartitionedCallStatefulPartitionedCalllambda_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_31957462
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
_user_specified_namelambda_6_input
�
�
*__inference_dense_19_layer_call_fn_3199137

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
E__inference_dense_19_layer_call_and_return_conditional_losses_31951922
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198465

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
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_3198014
lambda_6_input<
.batch_normalization_24_readvariableop_resource:>
0batch_normalization_24_readvariableop_1_resource:M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: <
.batch_normalization_25_readvariableop_resource: >
0batch_normalization_25_readvariableop_1_resource: M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: C
(conv2d_25_conv2d_readvariableop_resource: �8
)conv2d_25_biasadd_readvariableop_resource:	�=
.batch_normalization_26_readvariableop_resource:	�?
0batch_normalization_26_readvariableop_1_resource:	�N
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�=
.batch_normalization_27_readvariableop_resource:	�?
0batch_normalization_27_readvariableop_1_resource:	�N
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_27_conv2d_readvariableop_resource:��8
)conv2d_27_biasadd_readvariableop_resource:	�<
'dense_18_matmul_readvariableop_resource:��*�7
(dense_18_biasadd_readvariableop_resource:	�;
'dense_19_matmul_readvariableop_resource:
��7
(dense_19_biasadd_readvariableop_resource:	�
identity��6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_24/ReadVariableOp�'batch_normalization_24/ReadVariableOp_1�6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_25/ReadVariableOp�'batch_normalization_25/ReadVariableOp_1�6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_26/ReadVariableOp�'batch_normalization_26/ReadVariableOp_1�6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_27/ReadVariableOp�'batch_normalization_27/ReadVariableOp_1� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_6/strided_slice/stack�
lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_6/strided_slice/stack_1�
lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_6/strided_slice/stack_2�
lambda_6/strided_sliceStridedSlicelambda_6_input%lambda_6/strided_slice/stack:output:0'lambda_6/strided_slice/stack_1:output:0'lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_6/strided_slice�
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_24/ReadVariableOp�
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_24/ReadVariableOp_1�
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3lambda_6/strided_slice:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2)
'batch_normalization_24/FusedBatchNormV3�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_24/Conv2D�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_24/Relu�
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_25/ReadVariableOp�
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_25/ReadVariableOp_1�
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_24/Relu:activations:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_25/FusedBatchNormV3�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_25/Conv2D/ReadVariableOp�
conv2d_25/Conv2DConv2D+batch_normalization_25/FusedBatchNormV3:y:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
conv2d_25/Conv2D�
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp�
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
conv2d_25/Relu�
max_pooling2d_6/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool�
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_26/ReadVariableOp�
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_26/ReadVariableOp_1�
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_6/MaxPool:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_26/FusedBatchNormV3�
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_26/Conv2D/ReadVariableOp�
conv2d_26/Conv2DConv2D+batch_normalization_26/FusedBatchNormV3:y:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_26/Conv2D�
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp�
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_26/Relu�
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_27/ReadVariableOp�
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_27/ReadVariableOp_1�
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_26/Relu:activations:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3�
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2D+batch_normalization_27/FusedBatchNormV3:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_27/BiasAdd
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_27/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
flatten_6/Const�
flatten_6/ReshapeReshapeconv2d_27/Relu:activations:0flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2
flatten_6/Reshape�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulflatten_6/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_18/Relu�
dropout_12/IdentityIdentitydense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_12/Identity�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_12/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/BiasAddt
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_19/Relu�
dropout_13/IdentityIdentitydense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_13/Identity�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentitydropout_13/Identity:output:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_6_input
�
�
8__inference_batch_normalization_26_layer_call_fn_3198845

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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_31947512
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
�
�
%__inference_signature_wrapper_3196763
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
"__inference__wrapped_model_31944212
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
__inference_call_2942655

inputsI
;sequential_6_batch_normalization_24_readvariableop_resource:K
=sequential_6_batch_normalization_24_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_25_readvariableop_resource: K
=sequential_6_batch_normalization_25_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_6_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_25_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_26_readvariableop_resource:	�L
=sequential_6_batch_normalization_26_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_26_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_27_readvariableop_resource:	�L
=sequential_6_batch_normalization_27_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_27_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_27_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:��*�D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_24/ReadVariableOp�4sequential_6/batch_normalization_24/ReadVariableOp_1�Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_25/ReadVariableOp�4sequential_6/batch_normalization_25/ReadVariableOp_1�Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_26/ReadVariableOp�4sequential_6/batch_normalization_26/ReadVariableOp_1�Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_27/ReadVariableOp�4sequential_6/batch_normalization_27/ReadVariableOp_1�-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�,sequential_6/conv2d_24/Conv2D/ReadVariableOp�-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�,sequential_6/conv2d_25/Conv2D/ReadVariableOp�-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�,sequential_6/conv2d_26/Conv2D/ReadVariableOp�-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�,sequential_6/conv2d_27/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
)sequential_6/lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_6/lambda_6/strided_slice/stack�
+sequential_6/lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_6/lambda_6/strided_slice/stack_1�
+sequential_6/lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_6/lambda_6/strided_slice/stack_2�
#sequential_6/lambda_6/strided_sliceStridedSliceinputs2sequential_6/lambda_6/strided_slice/stack:output:04sequential_6/lambda_6/strided_slice/stack_1:output:04sequential_6/lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_6/lambda_6/strided_slice�
2sequential_6/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_6/batch_normalization_24/ReadVariableOp�
4sequential_6/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_6/batch_normalization_24/ReadVariableOp_1�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:0:sequential_6/batch_normalization_24/ReadVariableOp:value:0<sequential_6/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_24/FusedBatchNormV3�
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp�
sequential_6/conv2d_24/Conv2DConv2D8sequential_6/batch_normalization_24/FusedBatchNormV3:y:04sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D�
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_24/BiasAdd�
sequential_6/conv2d_24/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_24/Relu�
2sequential_6/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_6/batch_normalization_25/ReadVariableOp�
4sequential_6/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_6/batch_normalization_25/ReadVariableOp_1�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_24/Relu:activations:0:sequential_6/batch_normalization_25/ReadVariableOp:value:0<sequential_6/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_25/FusedBatchNormV3�
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp�
sequential_6/conv2d_25/Conv2DConv2D8sequential_6/batch_normalization_25/FusedBatchNormV3:y:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_6/conv2d_25/Conv2D�
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_6/conv2d_25/BiasAdd�
sequential_6/conv2d_25/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_6/conv2d_25/Relu�
$sequential_6/max_pooling2d_6/MaxPoolMaxPool)sequential_6/conv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_6/max_pooling2d_6/MaxPool�
2sequential_6/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_26/ReadVariableOp�
4sequential_6/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_26/ReadVariableOp_1�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3-sequential_6/max_pooling2d_6/MaxPool:output:0:sequential_6/batch_normalization_26/ReadVariableOp:value:0<sequential_6/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_26/FusedBatchNormV3�
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp�
sequential_6/conv2d_26/Conv2DConv2D8sequential_6/batch_normalization_26/FusedBatchNormV3:y:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D�
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_26/BiasAdd�
sequential_6/conv2d_26/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_26/Relu�
2sequential_6/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_27/ReadVariableOp�
4sequential_6/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_27/ReadVariableOp_1�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_26/Relu:activations:0:sequential_6/batch_normalization_27/ReadVariableOp:value:0<sequential_6/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_27/FusedBatchNormV3�
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp�
sequential_6/conv2d_27/Conv2DConv2D8sequential_6/batch_normalization_27/FusedBatchNormV3:y:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_27/Conv2D�
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_27/BiasAdd�
sequential_6/conv2d_27/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_27/Relu�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/conv2d_27/Relu:activations:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOp�
sequential_6/dense_18/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/MatMul�
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOp�
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/BiasAdd�
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/Relu�
 sequential_6/dropout_12/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_12/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_12/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/MatMul�
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOp�
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/BiasAdd�
sequential_6/dense_19/ReluRelu&sequential_6/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/Relu�
 sequential_6/dropout_13/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_13/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_13/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdd|
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_20/Softmax�
IdentityIdentitydense_20/Softmax:softmax:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpD^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_24/ReadVariableOp5^sequential_6/batch_normalization_24/ReadVariableOp_1D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_25/ReadVariableOp5^sequential_6/batch_normalization_25/ReadVariableOp_1D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_26/ReadVariableOp5^sequential_6/batch_normalization_26/ReadVariableOp_1D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_27/ReadVariableOp5^sequential_6/batch_normalization_27/ReadVariableOp_1.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_24/ReadVariableOp2sequential_6/batch_normalization_24/ReadVariableOp2l
4sequential_6/batch_normalization_24/ReadVariableOp_14sequential_6/batch_normalization_24/ReadVariableOp_12�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_25/ReadVariableOp2sequential_6/batch_normalization_25/ReadVariableOp2l
4sequential_6/batch_normalization_25/ReadVariableOp_14sequential_6/batch_normalization_25/ReadVariableOp_12�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_26/ReadVariableOp2sequential_6/batch_normalization_26/ReadVariableOp2l
4sequential_6/batch_normalization_26/ReadVariableOp_14sequential_6/batch_normalization_26/ReadVariableOp_12�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_27/ReadVariableOp2sequential_6/batch_normalization_27/ReadVariableOp2l
4sequential_6/batch_normalization_27/ReadVariableOp_14sequential_6/batch_normalization_27/ReadVariableOp_12^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3194751

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
&__inference_CNN3_layer_call_fn_3197530

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
A__inference_CNN3_layer_call_and_return_conditional_losses_31963762
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
�
M
1__inference_max_pooling2d_6_layer_call_fn_3194685

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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_31946792
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
�
�
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3194487

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
8__inference_batch_normalization_25_layer_call_fn_3198714

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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_31950212
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
�
�
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198657

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
�
�
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198639

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
8__inference_batch_normalization_26_layer_call_fn_3198858

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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_31950662
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3194877

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
�
�
E__inference_dense_19_layer_call_and_return_conditional_losses_3195192

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_12_layer_call_fn_3199100

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
GPU2*0J 8� *P
fKRI
G__inference_dropout_12_layer_call_and_return_conditional_losses_31951732
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
�
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3195110

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
�
�
E__inference_dense_19_layer_call_and_return_conditional_losses_3199128

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_3199197N
:dense_19_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_19/kernel/Regularizer/Square/ReadVariableOp�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_19_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentity#dense_19/kernel/Regularizer/mul:z:02^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp
�
�
F__inference_conv2d_24_layer_call_and_return_conditional_losses_3194998

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
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
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198621

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
�
f
G__inference_dropout_12_layer_call_and_return_conditional_losses_3195336

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
�
�
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3194569

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
�
G
+__inference_flatten_6_layer_call_fn_3199046

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
F__inference_flatten_6_layer_call_and_return_conditional_losses_31951432
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
�
F
*__inference_lambda_6_layer_call_fn_3198447

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
E__inference_lambda_6_layer_call_and_return_conditional_losses_31955872
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
�
�
+__inference_conv2d_27_layer_call_fn_3199035

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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_31951312
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
�
e
,__inference_dropout_13_layer_call_fn_3199164

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
GPU2*0J 8� *P
fKRI
G__inference_dropout_13_layer_call_and_return_conditional_losses_31953032
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
8__inference_batch_normalization_27_layer_call_fn_3198989

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_31948772
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
�
�
8__inference_batch_normalization_27_layer_call_fn_3199015

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_31953982
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
�
�
F__inference_conv2d_25_layer_call_and_return_conditional_losses_3195042

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
�
�
F__inference_conv2d_25_layer_call_and_return_conditional_losses_3198738

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
��
�!
A__inference_CNN3_layer_call_and_return_conditional_losses_3197185
input_1I
;sequential_6_batch_normalization_24_readvariableop_resource:K
=sequential_6_batch_normalization_24_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_25_readvariableop_resource: K
=sequential_6_batch_normalization_25_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_6_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_25_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_26_readvariableop_resource:	�L
=sequential_6_batch_normalization_26_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_26_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_27_readvariableop_resource:	�L
=sequential_6_batch_normalization_27_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_27_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_27_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:��*�D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_24/ReadVariableOp�4sequential_6/batch_normalization_24/ReadVariableOp_1�Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_25/ReadVariableOp�4sequential_6/batch_normalization_25/ReadVariableOp_1�Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_26/ReadVariableOp�4sequential_6/batch_normalization_26/ReadVariableOp_1�Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_27/ReadVariableOp�4sequential_6/batch_normalization_27/ReadVariableOp_1�-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�,sequential_6/conv2d_24/Conv2D/ReadVariableOp�-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�,sequential_6/conv2d_25/Conv2D/ReadVariableOp�-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�,sequential_6/conv2d_26/Conv2D/ReadVariableOp�-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�,sequential_6/conv2d_27/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
)sequential_6/lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_6/lambda_6/strided_slice/stack�
+sequential_6/lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_6/lambda_6/strided_slice/stack_1�
+sequential_6/lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_6/lambda_6/strided_slice/stack_2�
#sequential_6/lambda_6/strided_sliceStridedSliceinput_12sequential_6/lambda_6/strided_slice/stack:output:04sequential_6/lambda_6/strided_slice/stack_1:output:04sequential_6/lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_6/lambda_6/strided_slice�
2sequential_6/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_6/batch_normalization_24/ReadVariableOp�
4sequential_6/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_6/batch_normalization_24/ReadVariableOp_1�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:0:sequential_6/batch_normalization_24/ReadVariableOp:value:0<sequential_6/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_24/FusedBatchNormV3�
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp�
sequential_6/conv2d_24/Conv2DConv2D8sequential_6/batch_normalization_24/FusedBatchNormV3:y:04sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D�
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_24/BiasAdd�
sequential_6/conv2d_24/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_24/Relu�
2sequential_6/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_6/batch_normalization_25/ReadVariableOp�
4sequential_6/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_6/batch_normalization_25/ReadVariableOp_1�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_24/Relu:activations:0:sequential_6/batch_normalization_25/ReadVariableOp:value:0<sequential_6/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_25/FusedBatchNormV3�
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp�
sequential_6/conv2d_25/Conv2DConv2D8sequential_6/batch_normalization_25/FusedBatchNormV3:y:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_6/conv2d_25/Conv2D�
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_6/conv2d_25/BiasAdd�
sequential_6/conv2d_25/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_6/conv2d_25/Relu�
$sequential_6/max_pooling2d_6/MaxPoolMaxPool)sequential_6/conv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_6/max_pooling2d_6/MaxPool�
2sequential_6/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_26/ReadVariableOp�
4sequential_6/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_26/ReadVariableOp_1�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3-sequential_6/max_pooling2d_6/MaxPool:output:0:sequential_6/batch_normalization_26/ReadVariableOp:value:0<sequential_6/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_26/FusedBatchNormV3�
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp�
sequential_6/conv2d_26/Conv2DConv2D8sequential_6/batch_normalization_26/FusedBatchNormV3:y:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D�
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_26/BiasAdd�
sequential_6/conv2d_26/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_26/Relu�
2sequential_6/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_27/ReadVariableOp�
4sequential_6/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_27/ReadVariableOp_1�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_26/Relu:activations:0:sequential_6/batch_normalization_27/ReadVariableOp:value:0<sequential_6/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_27/FusedBatchNormV3�
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp�
sequential_6/conv2d_27/Conv2DConv2D8sequential_6/batch_normalization_27/FusedBatchNormV3:y:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_27/Conv2D�
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_27/BiasAdd�
sequential_6/conv2d_27/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_27/Relu�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/conv2d_27/Relu:activations:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOp�
sequential_6/dense_18/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/MatMul�
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOp�
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/BiasAdd�
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/Relu�
 sequential_6/dropout_12/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_12/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_12/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/MatMul�
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOp�
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/BiasAdd�
sequential_6/dense_19/ReluRelu&sequential_6/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/Relu�
 sequential_6/dropout_13/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_13/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_13/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdd|
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_20/Softmax�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentitydense_20/Softmax:softmax:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpD^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_24/ReadVariableOp5^sequential_6/batch_normalization_24/ReadVariableOp_1D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_25/ReadVariableOp5^sequential_6/batch_normalization_25/ReadVariableOp_1D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_26/ReadVariableOp5^sequential_6/batch_normalization_26/ReadVariableOp_1D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_27/ReadVariableOp5^sequential_6/batch_normalization_27/ReadVariableOp_1.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_24/ReadVariableOp2sequential_6/batch_normalization_24/ReadVariableOp2l
4sequential_6/batch_normalization_24/ReadVariableOp_14sequential_6/batch_normalization_24/ReadVariableOp_12�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_25/ReadVariableOp2sequential_6/batch_normalization_25/ReadVariableOp2l
4sequential_6/batch_normalization_25/ReadVariableOp_14sequential_6/batch_normalization_25/ReadVariableOp_12�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_26/ReadVariableOp2sequential_6/batch_normalization_26/ReadVariableOp2l
4sequential_6/batch_normalization_26/ReadVariableOp_14sequential_6/batch_normalization_26/ReadVariableOp_12�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_27/ReadVariableOp2sequential_6/batch_normalization_27/ReadVariableOp2l
4sequential_6/batch_normalization_27/ReadVariableOp_14sequential_6/batch_normalization_27/ReadVariableOp_12^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�r
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_3195746

inputs,
batch_normalization_24_3195657:,
batch_normalization_24_3195659:,
batch_normalization_24_3195661:,
batch_normalization_24_3195663:+
conv2d_24_3195666: 
conv2d_24_3195668: ,
batch_normalization_25_3195671: ,
batch_normalization_25_3195673: ,
batch_normalization_25_3195675: ,
batch_normalization_25_3195677: ,
conv2d_25_3195680: � 
conv2d_25_3195682:	�-
batch_normalization_26_3195686:	�-
batch_normalization_26_3195688:	�-
batch_normalization_26_3195690:	�-
batch_normalization_26_3195692:	�-
conv2d_26_3195695:�� 
conv2d_26_3195697:	�-
batch_normalization_27_3195700:	�-
batch_normalization_27_3195702:	�-
batch_normalization_27_3195704:	�-
batch_normalization_27_3195706:	�-
conv2d_27_3195709:�� 
conv2d_27_3195711:	�%
dense_18_3195715:��*�
dense_18_3195717:	�$
dense_19_3195721:
��
dense_19_3195723:	�
identity��.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�.batch_normalization_26/StatefulPartitionedCall�.batch_normalization_27/StatefulPartitionedCall�!conv2d_24/StatefulPartitionedCall�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�!conv2d_25/StatefulPartitionedCall�!conv2d_26/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall� dense_18/StatefulPartitionedCall�1dense_18/kernel/Regularizer/Square/ReadVariableOp� dense_19/StatefulPartitionedCall�1dense_19/kernel/Regularizer/Square/ReadVariableOp�"dropout_12/StatefulPartitionedCall�"dropout_13/StatefulPartitionedCall�
lambda_6/PartitionedCallPartitionedCallinputs*
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
E__inference_lambda_6_layer_call_and_return_conditional_losses_31955872
lambda_6/PartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0batch_normalization_24_3195657batch_normalization_24_3195659batch_normalization_24_3195661batch_normalization_24_3195663*
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_319556020
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_24_3195666conv2d_24_3195668*
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
F__inference_conv2d_24_layer_call_and_return_conditional_losses_31949982#
!conv2d_24/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_25_3195671batch_normalization_25_3195673batch_normalization_25_3195675batch_normalization_25_3195677*
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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_319550620
.batch_normalization_25/StatefulPartitionedCall�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_25_3195680conv2d_25_3195682*
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
F__inference_conv2d_25_layer_call_and_return_conditional_losses_31950422#
!conv2d_25/StatefulPartitionedCall�
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_31946792!
max_pooling2d_6/PartitionedCall�
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0batch_normalization_26_3195686batch_normalization_26_3195688batch_normalization_26_3195690batch_normalization_26_3195692*
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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_319545220
.batch_normalization_26/StatefulPartitionedCall�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_26_3195695conv2d_26_3195697*
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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_31950872#
!conv2d_26/StatefulPartitionedCall�
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_27_3195700batch_normalization_27_3195702batch_normalization_27_3195704batch_normalization_27_3195706*
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_319539820
.batch_normalization_27/StatefulPartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_27_3195709conv2d_27_3195711*
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_31951312#
!conv2d_27/StatefulPartitionedCall�
flatten_6/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
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
F__inference_flatten_6_layer_call_and_return_conditional_losses_31951432
flatten_6/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_18_3195715dense_18_3195717*
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
E__inference_dense_18_layer_call_and_return_conditional_losses_31951622"
 dense_18/StatefulPartitionedCall�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *P
fKRI
G__inference_dropout_12_layer_call_and_return_conditional_losses_31953362$
"dropout_12/StatefulPartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_19_3195721dense_19_3195723*
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
E__inference_dense_19_layer_call_and_return_conditional_losses_31951922"
 dense_19/StatefulPartitionedCall�
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
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
GPU2*0J 8� *P
fKRI
G__inference_dropout_13_layer_call_and_return_conditional_losses_31953032$
"dropout_13/StatefulPartitionedCall�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_3195666*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_3195715*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_3195721* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentity+dropout_13/StatefulPartitionedCall:output:0/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall2^dense_19/kernel/Regularizer/Square/ReadVariableOp#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
.__inference_sequential_6_layer_call_fn_3198279

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
I__inference_sequential_6_layer_call_and_return_conditional_losses_31952242
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
�
�
"__inference__wrapped_model_3194421
input_1
cnn3_3194359:
cnn3_3194361:
cnn3_3194363:
cnn3_3194365:&
cnn3_3194367: 
cnn3_3194369: 
cnn3_3194371: 
cnn3_3194373: 
cnn3_3194375: 
cnn3_3194377: '
cnn3_3194379: �
cnn3_3194381:	�
cnn3_3194383:	�
cnn3_3194385:	�
cnn3_3194387:	�
cnn3_3194389:	�(
cnn3_3194391:��
cnn3_3194393:	�
cnn3_3194395:	�
cnn3_3194397:	�
cnn3_3194399:	�
cnn3_3194401:	�(
cnn3_3194403:��
cnn3_3194405:	�!
cnn3_3194407:��*�
cnn3_3194409:	� 
cnn3_3194411:
��
cnn3_3194413:	�
cnn3_3194415:	�
cnn3_3194417:
identity��CNN3/StatefulPartitionedCall�
CNN3/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn3_3194359cnn3_3194361cnn3_3194363cnn3_3194365cnn3_3194367cnn3_3194369cnn3_3194371cnn3_3194373cnn3_3194375cnn3_3194377cnn3_3194379cnn3_3194381cnn3_3194383cnn3_3194385cnn3_3194387cnn3_3194389cnn3_3194391cnn3_3194393cnn3_3194395cnn3_3194397cnn3_3194399cnn3_3194401cnn3_3194403cnn3_3194405cnn3_3194407cnn3_3194409cnn3_3194411cnn3_3194413cnn3_3194415cnn3_3194417**
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
__inference_call_29390642
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
��
�$
A__inference_CNN3_layer_call_and_return_conditional_losses_3197049

inputsI
;sequential_6_batch_normalization_24_readvariableop_resource:K
=sequential_6_batch_normalization_24_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_25_readvariableop_resource: K
=sequential_6_batch_normalization_25_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_6_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_25_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_26_readvariableop_resource:	�L
=sequential_6_batch_normalization_26_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_26_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_27_readvariableop_resource:	�L
=sequential_6_batch_normalization_27_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_27_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_27_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:��*�D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�2sequential_6/batch_normalization_24/AssignNewValue�4sequential_6/batch_normalization_24/AssignNewValue_1�Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_24/ReadVariableOp�4sequential_6/batch_normalization_24/ReadVariableOp_1�2sequential_6/batch_normalization_25/AssignNewValue�4sequential_6/batch_normalization_25/AssignNewValue_1�Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_25/ReadVariableOp�4sequential_6/batch_normalization_25/ReadVariableOp_1�2sequential_6/batch_normalization_26/AssignNewValue�4sequential_6/batch_normalization_26/AssignNewValue_1�Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_26/ReadVariableOp�4sequential_6/batch_normalization_26/ReadVariableOp_1�2sequential_6/batch_normalization_27/AssignNewValue�4sequential_6/batch_normalization_27/AssignNewValue_1�Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_27/ReadVariableOp�4sequential_6/batch_normalization_27/ReadVariableOp_1�-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�,sequential_6/conv2d_24/Conv2D/ReadVariableOp�-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�,sequential_6/conv2d_25/Conv2D/ReadVariableOp�-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�,sequential_6/conv2d_26/Conv2D/ReadVariableOp�-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�,sequential_6/conv2d_27/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
)sequential_6/lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_6/lambda_6/strided_slice/stack�
+sequential_6/lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_6/lambda_6/strided_slice/stack_1�
+sequential_6/lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_6/lambda_6/strided_slice/stack_2�
#sequential_6/lambda_6/strided_sliceStridedSliceinputs2sequential_6/lambda_6/strided_slice/stack:output:04sequential_6/lambda_6/strided_slice/stack_1:output:04sequential_6/lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_6/lambda_6/strided_slice�
2sequential_6/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_6/batch_normalization_24/ReadVariableOp�
4sequential_6/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_6/batch_normalization_24/ReadVariableOp_1�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:0:sequential_6/batch_normalization_24/ReadVariableOp:value:0<sequential_6/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_6/batch_normalization_24/FusedBatchNormV3�
2sequential_6/batch_normalization_24/AssignNewValueAssignVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceAsequential_6/batch_normalization_24/FusedBatchNormV3:batch_mean:0D^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_6/batch_normalization_24/AssignNewValue�
4sequential_6/batch_normalization_24/AssignNewValue_1AssignVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resourceEsequential_6/batch_normalization_24/FusedBatchNormV3:batch_variance:0F^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_6/batch_normalization_24/AssignNewValue_1�
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp�
sequential_6/conv2d_24/Conv2DConv2D8sequential_6/batch_normalization_24/FusedBatchNormV3:y:04sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D�
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_24/BiasAdd�
sequential_6/conv2d_24/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_24/Relu�
2sequential_6/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_6/batch_normalization_25/ReadVariableOp�
4sequential_6/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_6/batch_normalization_25/ReadVariableOp_1�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_24/Relu:activations:0:sequential_6/batch_normalization_25/ReadVariableOp:value:0<sequential_6/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_6/batch_normalization_25/FusedBatchNormV3�
2sequential_6/batch_normalization_25/AssignNewValueAssignVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceAsequential_6/batch_normalization_25/FusedBatchNormV3:batch_mean:0D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_6/batch_normalization_25/AssignNewValue�
4sequential_6/batch_normalization_25/AssignNewValue_1AssignVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resourceEsequential_6/batch_normalization_25/FusedBatchNormV3:batch_variance:0F^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_6/batch_normalization_25/AssignNewValue_1�
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp�
sequential_6/conv2d_25/Conv2DConv2D8sequential_6/batch_normalization_25/FusedBatchNormV3:y:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_6/conv2d_25/Conv2D�
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_6/conv2d_25/BiasAdd�
sequential_6/conv2d_25/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_6/conv2d_25/Relu�
$sequential_6/max_pooling2d_6/MaxPoolMaxPool)sequential_6/conv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_6/max_pooling2d_6/MaxPool�
2sequential_6/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_26/ReadVariableOp�
4sequential_6/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_26/ReadVariableOp_1�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3-sequential_6/max_pooling2d_6/MaxPool:output:0:sequential_6/batch_normalization_26/ReadVariableOp:value:0<sequential_6/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_6/batch_normalization_26/FusedBatchNormV3�
2sequential_6/batch_normalization_26/AssignNewValueAssignVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceAsequential_6/batch_normalization_26/FusedBatchNormV3:batch_mean:0D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_6/batch_normalization_26/AssignNewValue�
4sequential_6/batch_normalization_26/AssignNewValue_1AssignVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceEsequential_6/batch_normalization_26/FusedBatchNormV3:batch_variance:0F^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_6/batch_normalization_26/AssignNewValue_1�
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp�
sequential_6/conv2d_26/Conv2DConv2D8sequential_6/batch_normalization_26/FusedBatchNormV3:y:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D�
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_26/BiasAdd�
sequential_6/conv2d_26/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_26/Relu�
2sequential_6/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_27/ReadVariableOp�
4sequential_6/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_27/ReadVariableOp_1�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_26/Relu:activations:0:sequential_6/batch_normalization_27/ReadVariableOp:value:0<sequential_6/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_6/batch_normalization_27/FusedBatchNormV3�
2sequential_6/batch_normalization_27/AssignNewValueAssignVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceAsequential_6/batch_normalization_27/FusedBatchNormV3:batch_mean:0D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_6/batch_normalization_27/AssignNewValue�
4sequential_6/batch_normalization_27/AssignNewValue_1AssignVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceEsequential_6/batch_normalization_27/FusedBatchNormV3:batch_variance:0F^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_6/batch_normalization_27/AssignNewValue_1�
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp�
sequential_6/conv2d_27/Conv2DConv2D8sequential_6/batch_normalization_27/FusedBatchNormV3:y:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_27/Conv2D�
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_27/BiasAdd�
sequential_6/conv2d_27/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_27/Relu�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/conv2d_27/Relu:activations:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOp�
sequential_6/dense_18/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/MatMul�
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOp�
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/BiasAdd�
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/Relu�
%sequential_6/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_6/dropout_12/dropout/Const�
#sequential_6/dropout_12/dropout/MulMul(sequential_6/dense_18/Relu:activations:0.sequential_6/dropout_12/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_6/dropout_12/dropout/Mul�
%sequential_6/dropout_12/dropout/ShapeShape(sequential_6/dense_18/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_12/dropout/Shape�
<sequential_6/dropout_12/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_12/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_6/dropout_12/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_6/dropout_12/dropout/GreaterEqual/y�
,sequential_6/dropout_12/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_12/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_12/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_6/dropout_12/dropout/GreaterEqual�
$sequential_6/dropout_12/dropout/CastCast0sequential_6/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_6/dropout_12/dropout/Cast�
%sequential_6/dropout_12/dropout/Mul_1Mul'sequential_6/dropout_12/dropout/Mul:z:0(sequential_6/dropout_12/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_6/dropout_12/dropout/Mul_1�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_12/dropout/Mul_1:z:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/MatMul�
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOp�
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/BiasAdd�
sequential_6/dense_19/ReluRelu&sequential_6/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/Relu�
%sequential_6/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_6/dropout_13/dropout/Const�
#sequential_6/dropout_13/dropout/MulMul(sequential_6/dense_19/Relu:activations:0.sequential_6/dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_6/dropout_13/dropout/Mul�
%sequential_6/dropout_13/dropout/ShapeShape(sequential_6/dense_19/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_13/dropout/Shape�
<sequential_6/dropout_13/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_6/dropout_13/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_6/dropout_13/dropout/GreaterEqual/y�
,sequential_6/dropout_13/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_13/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_6/dropout_13/dropout/GreaterEqual�
$sequential_6/dropout_13/dropout/CastCast0sequential_6/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_6/dropout_13/dropout/Cast�
%sequential_6/dropout_13/dropout/Mul_1Mul'sequential_6/dropout_13/dropout/Mul:z:0(sequential_6/dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_6/dropout_13/dropout/Mul_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_13/dropout/Mul_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdd|
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_20/Softmax�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentitydense_20/Softmax:softmax:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp3^sequential_6/batch_normalization_24/AssignNewValue5^sequential_6/batch_normalization_24/AssignNewValue_1D^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_24/ReadVariableOp5^sequential_6/batch_normalization_24/ReadVariableOp_13^sequential_6/batch_normalization_25/AssignNewValue5^sequential_6/batch_normalization_25/AssignNewValue_1D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_25/ReadVariableOp5^sequential_6/batch_normalization_25/ReadVariableOp_13^sequential_6/batch_normalization_26/AssignNewValue5^sequential_6/batch_normalization_26/AssignNewValue_1D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_26/ReadVariableOp5^sequential_6/batch_normalization_26/ReadVariableOp_13^sequential_6/batch_normalization_27/AssignNewValue5^sequential_6/batch_normalization_27/AssignNewValue_1D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_27/ReadVariableOp5^sequential_6/batch_normalization_27/ReadVariableOp_1.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2h
2sequential_6/batch_normalization_24/AssignNewValue2sequential_6/batch_normalization_24/AssignNewValue2l
4sequential_6/batch_normalization_24/AssignNewValue_14sequential_6/batch_normalization_24/AssignNewValue_12�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_24/ReadVariableOp2sequential_6/batch_normalization_24/ReadVariableOp2l
4sequential_6/batch_normalization_24/ReadVariableOp_14sequential_6/batch_normalization_24/ReadVariableOp_12h
2sequential_6/batch_normalization_25/AssignNewValue2sequential_6/batch_normalization_25/AssignNewValue2l
4sequential_6/batch_normalization_25/AssignNewValue_14sequential_6/batch_normalization_25/AssignNewValue_12�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_25/ReadVariableOp2sequential_6/batch_normalization_25/ReadVariableOp2l
4sequential_6/batch_normalization_25/ReadVariableOp_14sequential_6/batch_normalization_25/ReadVariableOp_12h
2sequential_6/batch_normalization_26/AssignNewValue2sequential_6/batch_normalization_26/AssignNewValue2l
4sequential_6/batch_normalization_26/AssignNewValue_14sequential_6/batch_normalization_26/AssignNewValue_12�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_26/ReadVariableOp2sequential_6/batch_normalization_26/ReadVariableOp2l
4sequential_6/batch_normalization_26/ReadVariableOp_14sequential_6/batch_normalization_26/ReadVariableOp_12h
2sequential_6/batch_normalization_27/AssignNewValue2sequential_6/batch_normalization_27/AssignNewValue2l
4sequential_6/batch_normalization_27/AssignNewValue_14sequential_6/batch_normalization_27/AssignNewValue_12�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_27/ReadVariableOp2sequential_6/batch_normalization_27/ReadVariableOp2l
4sequential_6/batch_normalization_27/ReadVariableOp_14sequential_6/batch_normalization_27/ReadVariableOp_12^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
��
�$
A__inference_CNN3_layer_call_and_return_conditional_losses_3197335
input_1I
;sequential_6_batch_normalization_24_readvariableop_resource:K
=sequential_6_batch_normalization_24_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_25_readvariableop_resource: K
=sequential_6_batch_normalization_25_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_6_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_25_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_26_readvariableop_resource:	�L
=sequential_6_batch_normalization_26_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_26_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_27_readvariableop_resource:	�L
=sequential_6_batch_normalization_27_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_27_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_27_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:��*�D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�2sequential_6/batch_normalization_24/AssignNewValue�4sequential_6/batch_normalization_24/AssignNewValue_1�Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_24/ReadVariableOp�4sequential_6/batch_normalization_24/ReadVariableOp_1�2sequential_6/batch_normalization_25/AssignNewValue�4sequential_6/batch_normalization_25/AssignNewValue_1�Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_25/ReadVariableOp�4sequential_6/batch_normalization_25/ReadVariableOp_1�2sequential_6/batch_normalization_26/AssignNewValue�4sequential_6/batch_normalization_26/AssignNewValue_1�Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_26/ReadVariableOp�4sequential_6/batch_normalization_26/ReadVariableOp_1�2sequential_6/batch_normalization_27/AssignNewValue�4sequential_6/batch_normalization_27/AssignNewValue_1�Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_27/ReadVariableOp�4sequential_6/batch_normalization_27/ReadVariableOp_1�-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�,sequential_6/conv2d_24/Conv2D/ReadVariableOp�-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�,sequential_6/conv2d_25/Conv2D/ReadVariableOp�-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�,sequential_6/conv2d_26/Conv2D/ReadVariableOp�-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�,sequential_6/conv2d_27/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
)sequential_6/lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_6/lambda_6/strided_slice/stack�
+sequential_6/lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_6/lambda_6/strided_slice/stack_1�
+sequential_6/lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_6/lambda_6/strided_slice/stack_2�
#sequential_6/lambda_6/strided_sliceStridedSliceinput_12sequential_6/lambda_6/strided_slice/stack:output:04sequential_6/lambda_6/strided_slice/stack_1:output:04sequential_6/lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_6/lambda_6/strided_slice�
2sequential_6/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_6/batch_normalization_24/ReadVariableOp�
4sequential_6/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_6/batch_normalization_24/ReadVariableOp_1�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:0:sequential_6/batch_normalization_24/ReadVariableOp:value:0<sequential_6/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_6/batch_normalization_24/FusedBatchNormV3�
2sequential_6/batch_normalization_24/AssignNewValueAssignVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceAsequential_6/batch_normalization_24/FusedBatchNormV3:batch_mean:0D^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_6/batch_normalization_24/AssignNewValue�
4sequential_6/batch_normalization_24/AssignNewValue_1AssignVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resourceEsequential_6/batch_normalization_24/FusedBatchNormV3:batch_variance:0F^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_6/batch_normalization_24/AssignNewValue_1�
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp�
sequential_6/conv2d_24/Conv2DConv2D8sequential_6/batch_normalization_24/FusedBatchNormV3:y:04sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D�
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_24/BiasAdd�
sequential_6/conv2d_24/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_24/Relu�
2sequential_6/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_6/batch_normalization_25/ReadVariableOp�
4sequential_6/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_6/batch_normalization_25/ReadVariableOp_1�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_24/Relu:activations:0:sequential_6/batch_normalization_25/ReadVariableOp:value:0<sequential_6/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_6/batch_normalization_25/FusedBatchNormV3�
2sequential_6/batch_normalization_25/AssignNewValueAssignVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceAsequential_6/batch_normalization_25/FusedBatchNormV3:batch_mean:0D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_6/batch_normalization_25/AssignNewValue�
4sequential_6/batch_normalization_25/AssignNewValue_1AssignVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resourceEsequential_6/batch_normalization_25/FusedBatchNormV3:batch_variance:0F^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_6/batch_normalization_25/AssignNewValue_1�
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp�
sequential_6/conv2d_25/Conv2DConv2D8sequential_6/batch_normalization_25/FusedBatchNormV3:y:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_6/conv2d_25/Conv2D�
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_6/conv2d_25/BiasAdd�
sequential_6/conv2d_25/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_6/conv2d_25/Relu�
$sequential_6/max_pooling2d_6/MaxPoolMaxPool)sequential_6/conv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_6/max_pooling2d_6/MaxPool�
2sequential_6/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_26/ReadVariableOp�
4sequential_6/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_26/ReadVariableOp_1�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3-sequential_6/max_pooling2d_6/MaxPool:output:0:sequential_6/batch_normalization_26/ReadVariableOp:value:0<sequential_6/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_6/batch_normalization_26/FusedBatchNormV3�
2sequential_6/batch_normalization_26/AssignNewValueAssignVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceAsequential_6/batch_normalization_26/FusedBatchNormV3:batch_mean:0D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_6/batch_normalization_26/AssignNewValue�
4sequential_6/batch_normalization_26/AssignNewValue_1AssignVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceEsequential_6/batch_normalization_26/FusedBatchNormV3:batch_variance:0F^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_6/batch_normalization_26/AssignNewValue_1�
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp�
sequential_6/conv2d_26/Conv2DConv2D8sequential_6/batch_normalization_26/FusedBatchNormV3:y:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D�
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_26/BiasAdd�
sequential_6/conv2d_26/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_26/Relu�
2sequential_6/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_27/ReadVariableOp�
4sequential_6/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_27/ReadVariableOp_1�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_26/Relu:activations:0:sequential_6/batch_normalization_27/ReadVariableOp:value:0<sequential_6/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<26
4sequential_6/batch_normalization_27/FusedBatchNormV3�
2sequential_6/batch_normalization_27/AssignNewValueAssignVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceAsequential_6/batch_normalization_27/FusedBatchNormV3:batch_mean:0D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential_6/batch_normalization_27/AssignNewValue�
4sequential_6/batch_normalization_27/AssignNewValue_1AssignVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceEsequential_6/batch_normalization_27/FusedBatchNormV3:batch_variance:0F^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype026
4sequential_6/batch_normalization_27/AssignNewValue_1�
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp�
sequential_6/conv2d_27/Conv2DConv2D8sequential_6/batch_normalization_27/FusedBatchNormV3:y:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_27/Conv2D�
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_27/BiasAdd�
sequential_6/conv2d_27/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_27/Relu�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/conv2d_27/Relu:activations:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOp�
sequential_6/dense_18/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/MatMul�
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOp�
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/BiasAdd�
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/Relu�
%sequential_6/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_6/dropout_12/dropout/Const�
#sequential_6/dropout_12/dropout/MulMul(sequential_6/dense_18/Relu:activations:0.sequential_6/dropout_12/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_6/dropout_12/dropout/Mul�
%sequential_6/dropout_12/dropout/ShapeShape(sequential_6/dense_18/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_12/dropout/Shape�
<sequential_6/dropout_12/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_12/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_6/dropout_12/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_6/dropout_12/dropout/GreaterEqual/y�
,sequential_6/dropout_12/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_12/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_12/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_6/dropout_12/dropout/GreaterEqual�
$sequential_6/dropout_12/dropout/CastCast0sequential_6/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_6/dropout_12/dropout/Cast�
%sequential_6/dropout_12/dropout/Mul_1Mul'sequential_6/dropout_12/dropout/Mul:z:0(sequential_6/dropout_12/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_6/dropout_12/dropout/Mul_1�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_12/dropout/Mul_1:z:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/MatMul�
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOp�
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/BiasAdd�
sequential_6/dense_19/ReluRelu&sequential_6/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/Relu�
%sequential_6/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_6/dropout_13/dropout/Const�
#sequential_6/dropout_13/dropout/MulMul(sequential_6/dense_19/Relu:activations:0.sequential_6/dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_6/dropout_13/dropout/Mul�
%sequential_6/dropout_13/dropout/ShapeShape(sequential_6/dense_19/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_13/dropout/Shape�
<sequential_6/dropout_13/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_6/dropout_13/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_6/dropout_13/dropout/GreaterEqual/y�
,sequential_6/dropout_13/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_13/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_6/dropout_13/dropout/GreaterEqual�
$sequential_6/dropout_13/dropout/CastCast0sequential_6/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_6/dropout_13/dropout/Cast�
%sequential_6/dropout_13/dropout/Mul_1Mul'sequential_6/dropout_13/dropout/Mul:z:0(sequential_6/dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_6/dropout_13/dropout/Mul_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_13/dropout/Mul_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdd|
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_20/Softmax�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentitydense_20/Softmax:softmax:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp3^sequential_6/batch_normalization_24/AssignNewValue5^sequential_6/batch_normalization_24/AssignNewValue_1D^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_24/ReadVariableOp5^sequential_6/batch_normalization_24/ReadVariableOp_13^sequential_6/batch_normalization_25/AssignNewValue5^sequential_6/batch_normalization_25/AssignNewValue_1D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_25/ReadVariableOp5^sequential_6/batch_normalization_25/ReadVariableOp_13^sequential_6/batch_normalization_26/AssignNewValue5^sequential_6/batch_normalization_26/AssignNewValue_1D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_26/ReadVariableOp5^sequential_6/batch_normalization_26/ReadVariableOp_13^sequential_6/batch_normalization_27/AssignNewValue5^sequential_6/batch_normalization_27/AssignNewValue_1D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_27/ReadVariableOp5^sequential_6/batch_normalization_27/ReadVariableOp_1.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2h
2sequential_6/batch_normalization_24/AssignNewValue2sequential_6/batch_normalization_24/AssignNewValue2l
4sequential_6/batch_normalization_24/AssignNewValue_14sequential_6/batch_normalization_24/AssignNewValue_12�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_24/ReadVariableOp2sequential_6/batch_normalization_24/ReadVariableOp2l
4sequential_6/batch_normalization_24/ReadVariableOp_14sequential_6/batch_normalization_24/ReadVariableOp_12h
2sequential_6/batch_normalization_25/AssignNewValue2sequential_6/batch_normalization_25/AssignNewValue2l
4sequential_6/batch_normalization_25/AssignNewValue_14sequential_6/batch_normalization_25/AssignNewValue_12�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_25/ReadVariableOp2sequential_6/batch_normalization_25/ReadVariableOp2l
4sequential_6/batch_normalization_25/ReadVariableOp_14sequential_6/batch_normalization_25/ReadVariableOp_12h
2sequential_6/batch_normalization_26/AssignNewValue2sequential_6/batch_normalization_26/AssignNewValue2l
4sequential_6/batch_normalization_26/AssignNewValue_14sequential_6/batch_normalization_26/AssignNewValue_12�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_26/ReadVariableOp2sequential_6/batch_normalization_26/ReadVariableOp2l
4sequential_6/batch_normalization_26/ReadVariableOp_14sequential_6/batch_normalization_26/ReadVariableOp_12h
2sequential_6/batch_normalization_27/AssignNewValue2sequential_6/batch_normalization_27/AssignNewValue2l
4sequential_6/batch_normalization_27/AssignNewValue_14sequential_6/batch_normalization_27/AssignNewValue_12�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_27/ReadVariableOp2sequential_6/batch_normalization_27/ReadVariableOp2l
4sequential_6/batch_normalization_27/ReadVariableOp_14sequential_6/batch_normalization_27/ReadVariableOp_12^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198675

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
�
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198909

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
�
�
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3195560

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
�
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_3195143

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
�
F__inference_conv2d_26_layer_call_and_return_conditional_losses_3195087

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
�
�
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198483

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
�
�
__inference_loss_fn_0_3199175U
;conv2d_24_kernel_regularizer_square_readvariableop_resource: 
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_24_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
IdentityIdentity$conv2d_24/kernel/Regularizer/mul:z:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp
�
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_3197742

inputs<
.batch_normalization_24_readvariableop_resource:>
0batch_normalization_24_readvariableop_1_resource:M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: <
.batch_normalization_25_readvariableop_resource: >
0batch_normalization_25_readvariableop_1_resource: M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: C
(conv2d_25_conv2d_readvariableop_resource: �8
)conv2d_25_biasadd_readvariableop_resource:	�=
.batch_normalization_26_readvariableop_resource:	�?
0batch_normalization_26_readvariableop_1_resource:	�N
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�=
.batch_normalization_27_readvariableop_resource:	�?
0batch_normalization_27_readvariableop_1_resource:	�N
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_27_conv2d_readvariableop_resource:��8
)conv2d_27_biasadd_readvariableop_resource:	�<
'dense_18_matmul_readvariableop_resource:��*�7
(dense_18_biasadd_readvariableop_resource:	�;
'dense_19_matmul_readvariableop_resource:
��7
(dense_19_biasadd_readvariableop_resource:	�
identity��6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_24/ReadVariableOp�'batch_normalization_24/ReadVariableOp_1�6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_25/ReadVariableOp�'batch_normalization_25/ReadVariableOp_1�6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_26/ReadVariableOp�'batch_normalization_26/ReadVariableOp_1�6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_27/ReadVariableOp�'batch_normalization_27/ReadVariableOp_1� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_6/strided_slice/stack�
lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_6/strided_slice/stack_1�
lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_6/strided_slice/stack_2�
lambda_6/strided_sliceStridedSliceinputs%lambda_6/strided_slice/stack:output:0'lambda_6/strided_slice/stack_1:output:0'lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_6/strided_slice�
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_24/ReadVariableOp�
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_24/ReadVariableOp_1�
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3lambda_6/strided_slice:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2)
'batch_normalization_24/FusedBatchNormV3�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_24/Conv2D�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_24/Relu�
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_25/ReadVariableOp�
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_25/ReadVariableOp_1�
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_24/Relu:activations:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_25/FusedBatchNormV3�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_25/Conv2D/ReadVariableOp�
conv2d_25/Conv2DConv2D+batch_normalization_25/FusedBatchNormV3:y:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
conv2d_25/Conv2D�
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp�
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
conv2d_25/Relu�
max_pooling2d_6/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool�
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_26/ReadVariableOp�
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_26/ReadVariableOp_1�
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_6/MaxPool:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_26/FusedBatchNormV3�
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_26/Conv2D/ReadVariableOp�
conv2d_26/Conv2DConv2D+batch_normalization_26/FusedBatchNormV3:y:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_26/Conv2D�
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp�
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_26/Relu�
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_27/ReadVariableOp�
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_27/ReadVariableOp_1�
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_26/Relu:activations:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3�
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2D+batch_normalization_27/FusedBatchNormV3:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_27/BiasAdd
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_27/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
flatten_6/Const�
flatten_6/ReshapeReshapeconv2d_27/Relu:activations:0flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2
flatten_6/Reshape�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulflatten_6/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_18/Relu�
dropout_12/IdentityIdentitydense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_12/Identity�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_12/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/BiasAddt
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_19/Relu�
dropout_13/IdentityIdentitydense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_13/Identity�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentitydropout_13/Identity:output:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
F__inference_conv2d_24_layer_call_and_return_conditional_losses_3198594

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
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
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_26_layer_call_fn_3198832

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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_31947072
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
�
�
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198765

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
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_3198412

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
�
�
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198819

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
�
�
+__inference_conv2d_24_layer_call_fn_3198603

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
F__inference_conv2d_24_layer_call_and_return_conditional_losses_31949982
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
��
�
__inference_call_2942419

inputsI
;sequential_6_batch_normalization_24_readvariableop_resource:K
=sequential_6_batch_normalization_24_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_25_readvariableop_resource: K
=sequential_6_batch_normalization_25_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_6_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_25_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_26_readvariableop_resource:	�L
=sequential_6_batch_normalization_26_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_26_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_27_readvariableop_resource:	�L
=sequential_6_batch_normalization_27_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_27_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_27_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:��*�D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_24/ReadVariableOp�4sequential_6/batch_normalization_24/ReadVariableOp_1�Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_25/ReadVariableOp�4sequential_6/batch_normalization_25/ReadVariableOp_1�Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_26/ReadVariableOp�4sequential_6/batch_normalization_26/ReadVariableOp_1�Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_27/ReadVariableOp�4sequential_6/batch_normalization_27/ReadVariableOp_1�-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�,sequential_6/conv2d_24/Conv2D/ReadVariableOp�-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�,sequential_6/conv2d_25/Conv2D/ReadVariableOp�-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�,sequential_6/conv2d_26/Conv2D/ReadVariableOp�-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�,sequential_6/conv2d_27/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
)sequential_6/lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_6/lambda_6/strided_slice/stack�
+sequential_6/lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_6/lambda_6/strided_slice/stack_1�
+sequential_6/lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_6/lambda_6/strided_slice/stack_2�
#sequential_6/lambda_6/strided_sliceStridedSliceinputs2sequential_6/lambda_6/strided_slice/stack:output:04sequential_6/lambda_6/strided_slice/stack_1:output:04sequential_6/lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_6/lambda_6/strided_slice�
2sequential_6/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_6/batch_normalization_24/ReadVariableOp�
4sequential_6/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_6/batch_normalization_24/ReadVariableOp_1�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:0:sequential_6/batch_normalization_24/ReadVariableOp:value:0<sequential_6/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_24/FusedBatchNormV3�
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp�
sequential_6/conv2d_24/Conv2DConv2D8sequential_6/batch_normalization_24/FusedBatchNormV3:y:04sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D�
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_6/conv2d_24/BiasAdd�
sequential_6/conv2d_24/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_6/conv2d_24/Relu�
2sequential_6/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_6/batch_normalization_25/ReadVariableOp�
4sequential_6/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_6/batch_normalization_25/ReadVariableOp_1�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_24/Relu:activations:0:sequential_6/batch_normalization_25/ReadVariableOp:value:0<sequential_6/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_25/FusedBatchNormV3�
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp�
sequential_6/conv2d_25/Conv2DConv2D8sequential_6/batch_normalization_25/FusedBatchNormV3:y:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�KK�*
paddingSAME*
strides
2
sequential_6/conv2d_25/Conv2D�
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�KK�2 
sequential_6/conv2d_25/BiasAdd�
sequential_6/conv2d_25/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*(
_output_shapes
:�KK�2
sequential_6/conv2d_25/Relu�
$sequential_6/max_pooling2d_6/MaxPoolMaxPool)sequential_6/conv2d_25/Relu:activations:0*(
_output_shapes
:�%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_6/max_pooling2d_6/MaxPool�
2sequential_6/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_26/ReadVariableOp�
4sequential_6/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_26/ReadVariableOp_1�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3-sequential_6/max_pooling2d_6/MaxPool:output:0:sequential_6/batch_normalization_26/ReadVariableOp:value:0<sequential_6/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*H
_output_shapes6
4:�%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_26/FusedBatchNormV3�
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp�
sequential_6/conv2d_26/Conv2DConv2D8sequential_6/batch_normalization_26/FusedBatchNormV3:y:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D�
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_6/conv2d_26/BiasAdd�
sequential_6/conv2d_26/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_6/conv2d_26/Relu�
2sequential_6/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_27/ReadVariableOp�
4sequential_6/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_27/ReadVariableOp_1�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_26/Relu:activations:0:sequential_6/batch_normalization_27/ReadVariableOp:value:0<sequential_6/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*H
_output_shapes6
4:�%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_27/FusedBatchNormV3�
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp�
sequential_6/conv2d_27/Conv2DConv2D8sequential_6/batch_normalization_27/FusedBatchNormV3:y:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_6/conv2d_27/Conv2D�
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_6/conv2d_27/BiasAdd�
sequential_6/conv2d_27/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_6/conv2d_27/Relu�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/conv2d_27/Relu:activations:0%sequential_6/flatten_6/Const:output:0*
T0*!
_output_shapes
:���*2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOp�
sequential_6/dense_18/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_18/MatMul�
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOp�
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_18/BiasAdd�
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_18/Relu�
 sequential_6/dropout_12/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_6/dropout_12/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_12/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_19/MatMul�
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOp�
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_19/BiasAdd�
sequential_6/dense_19/ReluRelu&sequential_6/dense_19/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_19/Relu�
 sequential_6/dropout_13/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_6/dropout_13/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_13/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_20/BiasAddt
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_20/Softmax�
IdentityIdentitydense_20/Softmax:softmax:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpD^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_24/ReadVariableOp5^sequential_6/batch_normalization_24/ReadVariableOp_1D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_25/ReadVariableOp5^sequential_6/batch_normalization_25/ReadVariableOp_1D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_26/ReadVariableOp5^sequential_6/batch_normalization_26/ReadVariableOp_1D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_27/ReadVariableOp5^sequential_6/batch_normalization_27/ReadVariableOp_1.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:�KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_24/ReadVariableOp2sequential_6/batch_normalization_24/ReadVariableOp2l
4sequential_6/batch_normalization_24/ReadVariableOp_14sequential_6/batch_normalization_24/ReadVariableOp_12�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_25/ReadVariableOp2sequential_6/batch_normalization_25/ReadVariableOp2l
4sequential_6/batch_normalization_25/ReadVariableOp_14sequential_6/batch_normalization_25/ReadVariableOp_12�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_26/ReadVariableOp2sequential_6/batch_normalization_26/ReadVariableOp2l
4sequential_6/batch_normalization_26/ReadVariableOp_14sequential_6/batch_normalization_26/ReadVariableOp_12�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_27/ReadVariableOp2sequential_6/batch_normalization_27/ReadVariableOp2l
4sequential_6/batch_normalization_27/ReadVariableOp_14sequential_6/batch_normalization_27/ReadVariableOp_12^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�9
�
A__inference_CNN3_layer_call_and_return_conditional_losses_3196152

inputs"
sequential_6_3196059:"
sequential_6_3196061:"
sequential_6_3196063:"
sequential_6_3196065:.
sequential_6_3196067: "
sequential_6_3196069: "
sequential_6_3196071: "
sequential_6_3196073: "
sequential_6_3196075: "
sequential_6_3196077: /
sequential_6_3196079: �#
sequential_6_3196081:	�#
sequential_6_3196083:	�#
sequential_6_3196085:	�#
sequential_6_3196087:	�#
sequential_6_3196089:	�0
sequential_6_3196091:��#
sequential_6_3196093:	�#
sequential_6_3196095:	�#
sequential_6_3196097:	�#
sequential_6_3196099:	�#
sequential_6_3196101:	�0
sequential_6_3196103:��#
sequential_6_3196105:	�)
sequential_6_3196107:��*�#
sequential_6_3196109:	�(
sequential_6_3196111:
��#
sequential_6_3196113:	�#
dense_20_3196128:	�
dense_20_3196130:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp� dense_20/StatefulPartitionedCall�$sequential_6/StatefulPartitionedCall�
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinputssequential_6_3196059sequential_6_3196061sequential_6_3196063sequential_6_3196065sequential_6_3196067sequential_6_3196069sequential_6_3196071sequential_6_3196073sequential_6_3196075sequential_6_3196077sequential_6_3196079sequential_6_3196081sequential_6_3196083sequential_6_3196085sequential_6_3196087sequential_6_3196089sequential_6_3196091sequential_6_3196093sequential_6_3196095sequential_6_3196097sequential_6_3196099sequential_6_3196101sequential_6_3196103sequential_6_3196105sequential_6_3196107sequential_6_3196109sequential_6_3196111sequential_6_3196113*(
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_31952242&
$sequential_6/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0dense_20_3196128dense_20_3196130*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_31961272"
 dense_20/StatefulPartitionedCall�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_3196067*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_3196107*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_3196111* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
��
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_3197885

inputs<
.batch_normalization_24_readvariableop_resource:>
0batch_normalization_24_readvariableop_1_resource:M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: <
.batch_normalization_25_readvariableop_resource: >
0batch_normalization_25_readvariableop_1_resource: M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: C
(conv2d_25_conv2d_readvariableop_resource: �8
)conv2d_25_biasadd_readvariableop_resource:	�=
.batch_normalization_26_readvariableop_resource:	�?
0batch_normalization_26_readvariableop_1_resource:	�N
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�=
.batch_normalization_27_readvariableop_resource:	�?
0batch_normalization_27_readvariableop_1_resource:	�N
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_27_conv2d_readvariableop_resource:��8
)conv2d_27_biasadd_readvariableop_resource:	�<
'dense_18_matmul_readvariableop_resource:��*�7
(dense_18_biasadd_readvariableop_resource:	�;
'dense_19_matmul_readvariableop_resource:
��7
(dense_19_biasadd_readvariableop_resource:	�
identity��%batch_normalization_24/AssignNewValue�'batch_normalization_24/AssignNewValue_1�6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_24/ReadVariableOp�'batch_normalization_24/ReadVariableOp_1�%batch_normalization_25/AssignNewValue�'batch_normalization_25/AssignNewValue_1�6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_25/ReadVariableOp�'batch_normalization_25/ReadVariableOp_1�%batch_normalization_26/AssignNewValue�'batch_normalization_26/AssignNewValue_1�6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_26/ReadVariableOp�'batch_normalization_26/ReadVariableOp_1�%batch_normalization_27/AssignNewValue�'batch_normalization_27/AssignNewValue_1�6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_27/ReadVariableOp�'batch_normalization_27/ReadVariableOp_1� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_6/strided_slice/stack�
lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_6/strided_slice/stack_1�
lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_6/strided_slice/stack_2�
lambda_6/strided_sliceStridedSliceinputs%lambda_6/strided_slice/stack:output:0'lambda_6/strided_slice/stack_1:output:0'lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_6/strided_slice�
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_24/ReadVariableOp�
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_24/ReadVariableOp_1�
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3lambda_6/strided_slice:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_24/FusedBatchNormV3�
%batch_normalization_24/AssignNewValueAssignVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource4batch_normalization_24/FusedBatchNormV3:batch_mean:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_24/AssignNewValue�
'batch_normalization_24/AssignNewValue_1AssignVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_24/FusedBatchNormV3:batch_variance:09^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_24/AssignNewValue_1�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_24/Conv2D�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_24/Relu�
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_25/ReadVariableOp�
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_25/ReadVariableOp_1�
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_24/Relu:activations:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_25/FusedBatchNormV3�
%batch_normalization_25/AssignNewValueAssignVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource4batch_normalization_25/FusedBatchNormV3:batch_mean:07^batch_normalization_25/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_25/AssignNewValue�
'batch_normalization_25/AssignNewValue_1AssignVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_25/FusedBatchNormV3:batch_variance:09^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_25/AssignNewValue_1�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_25/Conv2D/ReadVariableOp�
conv2d_25/Conv2DConv2D+batch_normalization_25/FusedBatchNormV3:y:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
conv2d_25/Conv2D�
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp�
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
conv2d_25/Relu�
max_pooling2d_6/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool�
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_26/ReadVariableOp�
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_26/ReadVariableOp_1�
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_6/MaxPool:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_26/FusedBatchNormV3�
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_26/AssignNewValue�
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_26/AssignNewValue_1�
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_26/Conv2D/ReadVariableOp�
conv2d_26/Conv2DConv2D+batch_normalization_26/FusedBatchNormV3:y:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_26/Conv2D�
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp�
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_26/Relu�
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_27/ReadVariableOp�
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_27/ReadVariableOp_1�
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_26/Relu:activations:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_27/FusedBatchNormV3�
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue�
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1�
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2D+batch_normalization_27/FusedBatchNormV3:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_27/BiasAdd
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_27/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
flatten_6/Const�
flatten_6/ReshapeReshapeconv2d_27/Relu:activations:0flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2
flatten_6/Reshape�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulflatten_6/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_18/Reluy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_12/dropout/Const�
dropout_12/dropout/MulMuldense_18/Relu:activations:0!dropout_12/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_12/dropout/Mul
dropout_12/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shape�
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform�
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_12/dropout/GreaterEqual/y�
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_12/dropout/GreaterEqual�
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_12/dropout/Cast�
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_12/dropout/Mul_1�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_12/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/BiasAddt
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_19/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Const�
dropout_13/dropout/MulMuldense_19/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape�
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform�
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/y�
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_13/dropout/GreaterEqual�
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_13/dropout/Cast�
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_13/dropout/Mul_1�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentitydropout_13/dropout/Mul_1:z:0&^batch_normalization_24/AssignNewValue(^batch_normalization_24/AssignNewValue_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_1&^batch_normalization_25/AssignNewValue(^batch_normalization_25/AssignNewValue_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_24/AssignNewValue%batch_normalization_24/AssignNewValue2R
'batch_normalization_24/AssignNewValue_1'batch_normalization_24/AssignNewValue_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12N
%batch_normalization_25/AssignNewValue%batch_normalization_25/AssignNewValue2R
'batch_normalization_25/AssignNewValue_1'batch_normalization_25/AssignNewValue_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3195066

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
�
e
G__inference_dropout_12_layer_call_and_return_conditional_losses_3195173

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
.__inference_sequential_6_layer_call_fn_3198340

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
I__inference_sequential_6_layer_call_and_return_conditional_losses_31957462
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
f
G__inference_dropout_13_layer_call_and_return_conditional_losses_3195303

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
�
�
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198501

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
�
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198927

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
�
�
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198519

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
��
�!
A__inference_CNN3_layer_call_and_return_conditional_losses_3196899

inputsI
;sequential_6_batch_normalization_24_readvariableop_resource:K
=sequential_6_batch_normalization_24_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_25_readvariableop_resource: K
=sequential_6_batch_normalization_25_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_6_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_25_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_26_readvariableop_resource:	�L
=sequential_6_batch_normalization_26_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_26_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_27_readvariableop_resource:	�L
=sequential_6_batch_normalization_27_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_27_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_27_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:��*�D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_24/ReadVariableOp�4sequential_6/batch_normalization_24/ReadVariableOp_1�Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_25/ReadVariableOp�4sequential_6/batch_normalization_25/ReadVariableOp_1�Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_26/ReadVariableOp�4sequential_6/batch_normalization_26/ReadVariableOp_1�Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_27/ReadVariableOp�4sequential_6/batch_normalization_27/ReadVariableOp_1�-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�,sequential_6/conv2d_24/Conv2D/ReadVariableOp�-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�,sequential_6/conv2d_25/Conv2D/ReadVariableOp�-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�,sequential_6/conv2d_26/Conv2D/ReadVariableOp�-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�,sequential_6/conv2d_27/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
)sequential_6/lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_6/lambda_6/strided_slice/stack�
+sequential_6/lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_6/lambda_6/strided_slice/stack_1�
+sequential_6/lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_6/lambda_6/strided_slice/stack_2�
#sequential_6/lambda_6/strided_sliceStridedSliceinputs2sequential_6/lambda_6/strided_slice/stack:output:04sequential_6/lambda_6/strided_slice/stack_1:output:04sequential_6/lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_6/lambda_6/strided_slice�
2sequential_6/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_6/batch_normalization_24/ReadVariableOp�
4sequential_6/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_6/batch_normalization_24/ReadVariableOp_1�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:0:sequential_6/batch_normalization_24/ReadVariableOp:value:0<sequential_6/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_24/FusedBatchNormV3�
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp�
sequential_6/conv2d_24/Conv2DConv2D8sequential_6/batch_normalization_24/FusedBatchNormV3:y:04sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D�
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_24/BiasAdd�
sequential_6/conv2d_24/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_24/Relu�
2sequential_6/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_6/batch_normalization_25/ReadVariableOp�
4sequential_6/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_6/batch_normalization_25/ReadVariableOp_1�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_24/Relu:activations:0:sequential_6/batch_normalization_25/ReadVariableOp:value:0<sequential_6/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_25/FusedBatchNormV3�
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp�
sequential_6/conv2d_25/Conv2DConv2D8sequential_6/batch_normalization_25/FusedBatchNormV3:y:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
2
sequential_6/conv2d_25/Conv2D�
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�2 
sequential_6/conv2d_25/BiasAdd�
sequential_6/conv2d_25/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�2
sequential_6/conv2d_25/Relu�
$sequential_6/max_pooling2d_6/MaxPoolMaxPool)sequential_6/conv2d_25/Relu:activations:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_6/max_pooling2d_6/MaxPool�
2sequential_6/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_26/ReadVariableOp�
4sequential_6/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_26/ReadVariableOp_1�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3-sequential_6/max_pooling2d_6/MaxPool:output:0:sequential_6/batch_normalization_26/ReadVariableOp:value:0<sequential_6/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_26/FusedBatchNormV3�
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp�
sequential_6/conv2d_26/Conv2DConv2D8sequential_6/batch_normalization_26/FusedBatchNormV3:y:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D�
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_26/BiasAdd�
sequential_6/conv2d_26/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_26/Relu�
2sequential_6/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_27/ReadVariableOp�
4sequential_6/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_27/ReadVariableOp_1�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_26/Relu:activations:0:sequential_6/batch_normalization_27/ReadVariableOp:value:0<sequential_6/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_27/FusedBatchNormV3�
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp�
sequential_6/conv2d_27/Conv2DConv2D8sequential_6/batch_normalization_27/FusedBatchNormV3:y:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_27/Conv2D�
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_27/BiasAdd�
sequential_6/conv2d_27/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_27/Relu�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/conv2d_27/Relu:activations:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������*2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOp�
sequential_6/dense_18/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/MatMul�
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOp�
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/BiasAdd�
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_18/Relu�
 sequential_6/dropout_12/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_12/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_12/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/MatMul�
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOp�
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/BiasAdd�
sequential_6/dense_19/ReluRelu&sequential_6/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_6/dense_19/Relu�
 sequential_6/dropout_13/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_13/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_13/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdd|
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_20/Softmax�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentitydense_20/Softmax:softmax:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpD^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_24/ReadVariableOp5^sequential_6/batch_normalization_24/ReadVariableOp_1D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_25/ReadVariableOp5^sequential_6/batch_normalization_25/ReadVariableOp_1D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_26/ReadVariableOp5^sequential_6/batch_normalization_26/ReadVariableOp_1D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_27/ReadVariableOp5^sequential_6/batch_normalization_27/ReadVariableOp_1.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_24/ReadVariableOp2sequential_6/batch_normalization_24/ReadVariableOp2l
4sequential_6/batch_normalization_24/ReadVariableOp_14sequential_6/batch_normalization_24/ReadVariableOp_12�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_25/ReadVariableOp2sequential_6/batch_normalization_25/ReadVariableOp2l
4sequential_6/batch_normalization_25/ReadVariableOp_14sequential_6/batch_normalization_25/ReadVariableOp_12�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_26/ReadVariableOp2sequential_6/batch_normalization_26/ReadVariableOp2l
4sequential_6/batch_normalization_26/ReadVariableOp_14sequential_6/batch_normalization_26/ReadVariableOp_12�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_27/ReadVariableOp2sequential_6/batch_normalization_27/ReadVariableOp2l
4sequential_6/batch_normalization_27/ReadVariableOp_14sequential_6/batch_normalization_27/ReadVariableOp_12^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
+__inference_conv2d_25_layer_call_fn_3198747

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
F__inference_conv2d_25_layer_call_and_return_conditional_losses_31950422
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
�
�
F__inference_conv2d_27_layer_call_and_return_conditional_losses_3199026

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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3195021

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
�
�
E__inference_dense_18_layer_call_and_return_conditional_losses_3195162

inputs3
matmul_readvariableop_resource:��*�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������*
 
_user_specified_nameinputs
�
�
+__inference_conv2d_26_layer_call_fn_3198891

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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_31950872
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
�
�
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3195452

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
�
e
G__inference_dropout_13_layer_call_and_return_conditional_losses_3195203

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
��
�
__inference_call_2942537

inputsI
;sequential_6_batch_normalization_24_readvariableop_resource:K
=sequential_6_batch_normalization_24_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_25_readvariableop_resource: K
=sequential_6_batch_normalization_25_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: P
5sequential_6_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_25_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_26_readvariableop_resource:	�L
=sequential_6_batch_normalization_26_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_26_biasadd_readvariableop_resource:	�J
;sequential_6_batch_normalization_27_readvariableop_resource:	�L
=sequential_6_batch_normalization_27_readvariableop_1_resource:	�[
Lsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:	�]
Nsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:	�Q
5sequential_6_conv2d_27_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_27_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:��*�D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_24/ReadVariableOp�4sequential_6/batch_normalization_24/ReadVariableOp_1�Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_25/ReadVariableOp�4sequential_6/batch_normalization_25/ReadVariableOp_1�Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_26/ReadVariableOp�4sequential_6/batch_normalization_26/ReadVariableOp_1�Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�2sequential_6/batch_normalization_27/ReadVariableOp�4sequential_6/batch_normalization_27/ReadVariableOp_1�-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�,sequential_6/conv2d_24/Conv2D/ReadVariableOp�-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�,sequential_6/conv2d_25/Conv2D/ReadVariableOp�-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�,sequential_6/conv2d_26/Conv2D/ReadVariableOp�-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�,sequential_6/conv2d_27/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
)sequential_6/lambda_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_6/lambda_6/strided_slice/stack�
+sequential_6/lambda_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_6/lambda_6/strided_slice/stack_1�
+sequential_6/lambda_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_6/lambda_6/strided_slice/stack_2�
#sequential_6/lambda_6/strided_sliceStridedSliceinputs2sequential_6/lambda_6/strided_slice/stack:output:04sequential_6/lambda_6/strided_slice/stack_1:output:04sequential_6/lambda_6/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_6/lambda_6/strided_slice�
2sequential_6/batch_normalization_24/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_24_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_6/batch_normalization_24/ReadVariableOp�
4sequential_6/batch_normalization_24/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:*
dtype026
4sequential_6/batch_normalization_24/ReadVariableOp_1�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:0:sequential_6/batch_normalization_24/ReadVariableOp:value:0<sequential_6/batch_normalization_24/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_24/FusedBatchNormV3�
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp�
sequential_6/conv2d_24/Conv2DConv2D8sequential_6/batch_normalization_24/FusedBatchNormV3:y:04sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D�
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp�
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_6/conv2d_24/BiasAdd�
sequential_6/conv2d_24/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_6/conv2d_24/Relu�
2sequential_6/batch_normalization_25/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_6/batch_normalization_25/ReadVariableOp�
4sequential_6/batch_normalization_25/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_6/batch_normalization_25/ReadVariableOp_1�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_24/Relu:activations:0:sequential_6/batch_normalization_25/ReadVariableOp:value:0<sequential_6/batch_normalization_25/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK : : : : :*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_25/FusedBatchNormV3�
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp�
sequential_6/conv2d_25/Conv2DConv2D8sequential_6/batch_normalization_25/FusedBatchNormV3:y:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�KK�*
paddingSAME*
strides
2
sequential_6/conv2d_25/Conv2D�
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp�
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�KK�2 
sequential_6/conv2d_25/BiasAdd�
sequential_6/conv2d_25/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*(
_output_shapes
:�KK�2
sequential_6/conv2d_25/Relu�
$sequential_6/max_pooling2d_6/MaxPoolMaxPool)sequential_6/conv2d_25/Relu:activations:0*(
_output_shapes
:�%%�*
ksize
*
paddingVALID*
strides
2&
$sequential_6/max_pooling2d_6/MaxPool�
2sequential_6/batch_normalization_26/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_26_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_26/ReadVariableOp�
4sequential_6/batch_normalization_26/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_26/ReadVariableOp_1�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3-sequential_6/max_pooling2d_6/MaxPool:output:0:sequential_6/batch_normalization_26/ReadVariableOp:value:0<sequential_6/batch_normalization_26/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*H
_output_shapes6
4:�%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_26/FusedBatchNormV3�
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp�
sequential_6/conv2d_26/Conv2DConv2D8sequential_6/batch_normalization_26/FusedBatchNormV3:y:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D�
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp�
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_6/conv2d_26/BiasAdd�
sequential_6/conv2d_26/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_6/conv2d_26/Relu�
2sequential_6/batch_normalization_27/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_27_readvariableop_resource*
_output_shapes	
:�*
dtype024
2sequential_6/batch_normalization_27/ReadVariableOp�
4sequential_6/batch_normalization_27/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_27_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4sequential_6/batch_normalization_27/ReadVariableOp_1�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02E
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
4sequential_6/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3)sequential_6/conv2d_26/Relu:activations:0:sequential_6/batch_normalization_27/ReadVariableOp:value:0<sequential_6/batch_normalization_27/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*H
_output_shapes6
4:�%%�:�:�:�:�:*
epsilon%o�:*
is_training( 26
4sequential_6/batch_normalization_27/FusedBatchNormV3�
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp�
sequential_6/conv2d_27/Conv2DConv2D8sequential_6/batch_normalization_27/FusedBatchNormV3:y:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_6/conv2d_27/Conv2D�
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp�
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_6/conv2d_27/BiasAdd�
sequential_6/conv2d_27/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_6/conv2d_27/Relu�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �
 2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/conv2d_27/Relu:activations:0%sequential_6/flatten_6/Const:output:0*
T0*!
_output_shapes
:���*2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:��*�*
dtype02-
+sequential_6/dense_18/MatMul/ReadVariableOp�
sequential_6/dense_18/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_18/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_18/MatMul�
,sequential_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_18/BiasAdd/ReadVariableOp�
sequential_6/dense_18/BiasAddBiasAdd&sequential_6/dense_18/MatMul:product:04sequential_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_18/BiasAdd�
sequential_6/dense_18/ReluRelu&sequential_6/dense_18/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_18/Relu�
 sequential_6/dropout_12/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_6/dropout_12/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_12/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_19/MatMul�
,sequential_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_6/dense_19/BiasAdd/ReadVariableOp�
sequential_6/dense_19/BiasAddBiasAdd&sequential_6/dense_19/MatMul:product:04sequential_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_19/BiasAdd�
sequential_6/dense_19/ReluRelu&sequential_6/dense_19/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_6/dense_19/Relu�
 sequential_6/dropout_13/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_6/dropout_13/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_13/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_20/BiasAddt
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_20/Softmax�
IdentityIdentitydense_20/Softmax:softmax:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpD^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_24/ReadVariableOp5^sequential_6/batch_normalization_24/ReadVariableOp_1D^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_25/ReadVariableOp5^sequential_6/batch_normalization_25/ReadVariableOp_1D^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_26/ReadVariableOp5^sequential_6/batch_normalization_26/ReadVariableOp_1D^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_27/ReadVariableOp5^sequential_6/batch_normalization_27/ReadVariableOp_1.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:�KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Csequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_24/ReadVariableOp2sequential_6/batch_normalization_24/ReadVariableOp2l
4sequential_6/batch_normalization_24/ReadVariableOp_14sequential_6/batch_normalization_24/ReadVariableOp_12�
Csequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_25/ReadVariableOp2sequential_6/batch_normalization_25/ReadVariableOp2l
4sequential_6/batch_normalization_25/ReadVariableOp_14sequential_6/batch_normalization_25/ReadVariableOp_12�
Csequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_26/ReadVariableOp2sequential_6/batch_normalization_26/ReadVariableOp2l
4sequential_6/batch_normalization_26/ReadVariableOp_14sequential_6/batch_normalization_26/ReadVariableOp_12�
Csequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_27/ReadVariableOp2sequential_6/batch_normalization_27/ReadVariableOp2l
4sequential_6/batch_normalization_27/ReadVariableOp_14sequential_6/batch_normalization_27/ReadVariableOp_12^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
�
&__inference_CNN3_layer_call_fn_3197465

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
A__inference_CNN3_layer_call_and_return_conditional_losses_31961522
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
�
�
8__inference_batch_normalization_25_layer_call_fn_3198727

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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_31955062
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
�
�
8__inference_batch_normalization_24_layer_call_fn_3198571

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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_31955602
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
�9
�
A__inference_CNN3_layer_call_and_return_conditional_losses_3196376

inputs"
sequential_6_3196295:"
sequential_6_3196297:"
sequential_6_3196299:"
sequential_6_3196301:.
sequential_6_3196303: "
sequential_6_3196305: "
sequential_6_3196307: "
sequential_6_3196309: "
sequential_6_3196311: "
sequential_6_3196313: /
sequential_6_3196315: �#
sequential_6_3196317:	�#
sequential_6_3196319:	�#
sequential_6_3196321:	�#
sequential_6_3196323:	�#
sequential_6_3196325:	�0
sequential_6_3196327:��#
sequential_6_3196329:	�#
sequential_6_3196331:	�#
sequential_6_3196333:	�#
sequential_6_3196335:	�#
sequential_6_3196337:	�0
sequential_6_3196339:��#
sequential_6_3196341:	�)
sequential_6_3196343:��*�#
sequential_6_3196345:	�(
sequential_6_3196347:
��#
sequential_6_3196349:	�#
dense_20_3196352:	�
dense_20_3196354:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp� dense_20/StatefulPartitionedCall�$sequential_6/StatefulPartitionedCall�
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinputssequential_6_3196295sequential_6_3196297sequential_6_3196299sequential_6_3196301sequential_6_3196303sequential_6_3196305sequential_6_3196307sequential_6_3196309sequential_6_3196311sequential_6_3196313sequential_6_3196315sequential_6_3196317sequential_6_3196319sequential_6_3196321sequential_6_3196323sequential_6_3196325sequential_6_3196327sequential_6_3196329sequential_6_3196331sequential_6_3196333sequential_6_3196335sequential_6_3196337sequential_6_3196339sequential_6_3196341sequential_6_3196343sequential_6_3196345sequential_6_3196347sequential_6_3196349*(
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_31957462&
$sequential_6/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0dense_20_3196352dense_20_3196354*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_31961272"
 dense_20/StatefulPartitionedCall�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_3196303*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_3196343*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_3196347* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3195506

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
�
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_3195587

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
�
�
*__inference_dense_20_layer_call_fn_3198421

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
E__inference_dense_20_layer_call_and_return_conditional_losses_31961272
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
�
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3194679

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
�
�
8__inference_batch_normalization_26_layer_call_fn_3198871

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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_31954522
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
�
�
8__inference_batch_normalization_25_layer_call_fn_3198688

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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_31945692
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
�
�%
 __inference__traced_save_3199469
file_prefix.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_batch_normalization_24_gamma_read_readvariableop:
6savev2_batch_normalization_24_beta_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop;
7savev2_batch_normalization_25_gamma_read_readvariableop:
6savev2_batch_normalization_25_beta_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop;
7savev2_batch_normalization_26_gamma_read_readvariableop:
6savev2_batch_normalization_26_beta_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableopA
=savev2_batch_normalization_24_moving_mean_read_readvariableopE
Asavev2_batch_normalization_24_moving_variance_read_readvariableopA
=savev2_batch_normalization_25_moving_mean_read_readvariableopE
Asavev2_batch_normalization_25_moving_variance_read_readvariableopA
=savev2_batch_normalization_26_moving_mean_read_readvariableopE
Asavev2_batch_normalization_26_moving_variance_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_24_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_24_beta_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_25_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_25_beta_m_read_readvariableop6
2savev2_adam_conv2d_25_kernel_m_read_readvariableop4
0savev2_adam_conv2d_25_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_26_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_26_beta_m_read_readvariableop6
2savev2_adam_conv2d_26_kernel_m_read_readvariableop4
0savev2_adam_conv2d_26_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_27_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_27_beta_m_read_readvariableop6
2savev2_adam_conv2d_27_kernel_m_read_readvariableop4
0savev2_adam_conv2d_27_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_24_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_24_beta_v_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_25_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_25_beta_v_read_readvariableop6
2savev2_adam_conv2d_25_kernel_v_read_readvariableop4
0savev2_adam_conv2d_25_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_26_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_26_beta_v_read_readvariableop6
2savev2_adam_conv2d_26_kernel_v_read_readvariableop4
0savev2_adam_conv2d_26_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_27_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_27_beta_v_read_readvariableop6
2savev2_adam_conv2d_27_kernel_v_read_readvariableop4
0savev2_adam_conv2d_27_bias_v_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop
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
ShardedFilename�*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�)
value�)B�)TB)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�
value�B�TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_batch_normalization_24_gamma_read_readvariableop6savev2_batch_normalization_24_beta_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop7savev2_batch_normalization_25_gamma_read_readvariableop6savev2_batch_normalization_25_beta_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop7savev2_batch_normalization_26_gamma_read_readvariableop6savev2_batch_normalization_26_beta_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop=savev2_batch_normalization_24_moving_mean_read_readvariableopAsavev2_batch_normalization_24_moving_variance_read_readvariableop=savev2_batch_normalization_25_moving_mean_read_readvariableopAsavev2_batch_normalization_25_moving_variance_read_readvariableop=savev2_batch_normalization_26_moving_mean_read_readvariableopAsavev2_batch_normalization_26_moving_variance_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop>savev2_adam_batch_normalization_24_gamma_m_read_readvariableop=savev2_adam_batch_normalization_24_beta_m_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop>savev2_adam_batch_normalization_25_gamma_m_read_readvariableop=savev2_adam_batch_normalization_25_beta_m_read_readvariableop2savev2_adam_conv2d_25_kernel_m_read_readvariableop0savev2_adam_conv2d_25_bias_m_read_readvariableop>savev2_adam_batch_normalization_26_gamma_m_read_readvariableop=savev2_adam_batch_normalization_26_beta_m_read_readvariableop2savev2_adam_conv2d_26_kernel_m_read_readvariableop0savev2_adam_conv2d_26_bias_m_read_readvariableop>savev2_adam_batch_normalization_27_gamma_m_read_readvariableop=savev2_adam_batch_normalization_27_beta_m_read_readvariableop2savev2_adam_conv2d_27_kernel_m_read_readvariableop0savev2_adam_conv2d_27_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop>savev2_adam_batch_normalization_24_gamma_v_read_readvariableop=savev2_adam_batch_normalization_24_beta_v_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop>savev2_adam_batch_normalization_25_gamma_v_read_readvariableop=savev2_adam_batch_normalization_25_beta_v_read_readvariableop2savev2_adam_conv2d_25_kernel_v_read_readvariableop0savev2_adam_conv2d_25_bias_v_read_readvariableop>savev2_adam_batch_normalization_26_gamma_v_read_readvariableop=savev2_adam_batch_normalization_26_beta_v_read_readvariableop2savev2_adam_conv2d_26_kernel_v_read_readvariableop0savev2_adam_conv2d_26_bias_v_read_readvariableop>savev2_adam_batch_normalization_27_gamma_v_read_readvariableop=savev2_adam_batch_normalization_27_beta_v_read_readvariableop2savev2_adam_conv2d_27_kernel_v_read_readvariableop0savev2_adam_conv2d_27_bias_v_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�: :	�:: : : : : ::: : : : : �:�:�:�:��:�:�:�:��:�:��*�:�:
��:�::: : :�:�:�:�: : : : :	�:::: : : : : �:�:�:�:��:�:�:�:��:�:��*�:�:
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
::,
(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: �:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:'#
!
_output_shapes
:��*�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
: : 

_output_shapes
: :! 

_output_shapes	
:�:!!

_output_shapes	
:�:!"

_output_shapes	
:�:!#

_output_shapes	
:�:$
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
�
�
&__inference_CNN3_layer_call_fn_3197595
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
A__inference_CNN3_layer_call_and_return_conditional_losses_31963762
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
�
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3195398

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
�
�
*__inference_dense_18_layer_call_fn_3199078

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
E__inference_dense_18_layer_call_and_return_conditional_losses_31951622
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
�
�
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3194707

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
�
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198945

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
�
�
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3194613

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
�o
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_3195224

inputs,
batch_normalization_24_3194972:,
batch_normalization_24_3194974:,
batch_normalization_24_3194976:,
batch_normalization_24_3194978:+
conv2d_24_3194999: 
conv2d_24_3195001: ,
batch_normalization_25_3195022: ,
batch_normalization_25_3195024: ,
batch_normalization_25_3195026: ,
batch_normalization_25_3195028: ,
conv2d_25_3195043: � 
conv2d_25_3195045:	�-
batch_normalization_26_3195067:	�-
batch_normalization_26_3195069:	�-
batch_normalization_26_3195071:	�-
batch_normalization_26_3195073:	�-
conv2d_26_3195088:�� 
conv2d_26_3195090:	�-
batch_normalization_27_3195111:	�-
batch_normalization_27_3195113:	�-
batch_normalization_27_3195115:	�-
batch_normalization_27_3195117:	�-
conv2d_27_3195132:�� 
conv2d_27_3195134:	�%
dense_18_3195163:��*�
dense_18_3195165:	�$
dense_19_3195193:
��
dense_19_3195195:	�
identity��.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�.batch_normalization_26/StatefulPartitionedCall�.batch_normalization_27/StatefulPartitionedCall�!conv2d_24/StatefulPartitionedCall�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�!conv2d_25/StatefulPartitionedCall�!conv2d_26/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall� dense_18/StatefulPartitionedCall�1dense_18/kernel/Regularizer/Square/ReadVariableOp� dense_19/StatefulPartitionedCall�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
lambda_6/PartitionedCallPartitionedCallinputs*
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
E__inference_lambda_6_layer_call_and_return_conditional_losses_31949522
lambda_6/PartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0batch_normalization_24_3194972batch_normalization_24_3194974batch_normalization_24_3194976batch_normalization_24_3194978*
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_319497120
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_24_3194999conv2d_24_3195001*
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
F__inference_conv2d_24_layer_call_and_return_conditional_losses_31949982#
!conv2d_24/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_25_3195022batch_normalization_25_3195024batch_normalization_25_3195026batch_normalization_25_3195028*
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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_319502120
.batch_normalization_25/StatefulPartitionedCall�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_25_3195043conv2d_25_3195045*
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
F__inference_conv2d_25_layer_call_and_return_conditional_losses_31950422#
!conv2d_25/StatefulPartitionedCall�
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_31946792!
max_pooling2d_6/PartitionedCall�
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0batch_normalization_26_3195067batch_normalization_26_3195069batch_normalization_26_3195071batch_normalization_26_3195073*
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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_319506620
.batch_normalization_26/StatefulPartitionedCall�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_26_3195088conv2d_26_3195090*
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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_31950872#
!conv2d_26/StatefulPartitionedCall�
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_27_3195111batch_normalization_27_3195113batch_normalization_27_3195115batch_normalization_27_3195117*
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_319511020
.batch_normalization_27/StatefulPartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_27_3195132conv2d_27_3195134*
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_31951312#
!conv2d_27/StatefulPartitionedCall�
flatten_6/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
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
F__inference_flatten_6_layer_call_and_return_conditional_losses_31951432
flatten_6/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_18_3195163dense_18_3195165*
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
E__inference_dense_18_layer_call_and_return_conditional_losses_31951622"
 dense_18/StatefulPartitionedCall�
dropout_12/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *P
fKRI
G__inference_dropout_12_layer_call_and_return_conditional_losses_31951732
dropout_12/PartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_19_3195193dense_19_3195195*
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
E__inference_dense_19_layer_call_and_return_conditional_losses_31951922"
 dense_19/StatefulPartitionedCall�
dropout_13/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *P
fKRI
G__inference_dropout_13_layer_call_and_return_conditional_losses_31952032
dropout_13/PartitionedCall�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_3194999*&
_output_shapes
: *
dtype024
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_24/kernel/Regularizer/Square�
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_24/kernel/Regularizer/Const�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/Sum�
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_24/kernel/Regularizer/mul/x�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_24/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_3195163*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_3195193* 
_output_shapes
:
��*
dtype023
1dense_19/kernel/Regularizer/Square/ReadVariableOp�
"dense_19/kernel/Regularizer/SquareSquare9dense_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_19/kernel/Regularizer/Square�
!dense_19/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_19/kernel/Regularizer/Const�
dense_19/kernel/Regularizer/SumSum&dense_19/kernel/Regularizer/Square:y:0*dense_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/Sum�
!dense_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_19/kernel/Regularizer/mul/x�
dense_19/kernel/Regularizer/mulMul*dense_19/kernel/Regularizer/mul/x:output:0(dense_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_19/kernel/Regularizer/mul�
IdentityIdentity#dropout_13/PartitionedCall:output:0/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:���������KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
&__inference_CNN3_layer_call_fn_3197400
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
A__inference_CNN3_layer_call_and_return_conditional_losses_31961522
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
�
F__inference_conv2d_26_layer_call_and_return_conditional_losses_3198882

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
8__inference_batch_normalization_25_layer_call_fn_3198701

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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_31946132
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
E__inference_lambda_6_layer_call_and_return_conditional_losses_3198429

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
�
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_3199041

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
�
e
G__inference_dropout_12_layer_call_and_return_conditional_losses_3199083

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
�
__inference_loss_fn_1_3199186O
:dense_18_kernel_regularizer_square_readvariableop_resource:��*�
identity��1dense_18/kernel/Regularizer/Square/ReadVariableOp�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_18_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:��*�*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:��*�2$
"dense_18/kernel/Regularizer/Square�
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/Const�
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/Sum�
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_18/kernel/Regularizer/mul/x�
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul�
IdentityIdentity#dense_18/kernel/Regularizer/mul:z:02^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp
�
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_3194952

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
��
�6
#__inference__traced_restore_3199728
file_prefix3
 assignvariableop_dense_20_kernel:	�.
 assignvariableop_1_dense_20_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: =
/assignvariableop_7_batch_normalization_24_gamma:<
.assignvariableop_8_batch_normalization_24_beta:=
#assignvariableop_9_conv2d_24_kernel: 0
"assignvariableop_10_conv2d_24_bias: >
0assignvariableop_11_batch_normalization_25_gamma: =
/assignvariableop_12_batch_normalization_25_beta: ?
$assignvariableop_13_conv2d_25_kernel: �1
"assignvariableop_14_conv2d_25_bias:	�?
0assignvariableop_15_batch_normalization_26_gamma:	�>
/assignvariableop_16_batch_normalization_26_beta:	�@
$assignvariableop_17_conv2d_26_kernel:��1
"assignvariableop_18_conv2d_26_bias:	�?
0assignvariableop_19_batch_normalization_27_gamma:	�>
/assignvariableop_20_batch_normalization_27_beta:	�@
$assignvariableop_21_conv2d_27_kernel:��1
"assignvariableop_22_conv2d_27_bias:	�8
#assignvariableop_23_dense_18_kernel:��*�0
!assignvariableop_24_dense_18_bias:	�7
#assignvariableop_25_dense_19_kernel:
��0
!assignvariableop_26_dense_19_bias:	�D
6assignvariableop_27_batch_normalization_24_moving_mean:H
:assignvariableop_28_batch_normalization_24_moving_variance:D
6assignvariableop_29_batch_normalization_25_moving_mean: H
:assignvariableop_30_batch_normalization_25_moving_variance: E
6assignvariableop_31_batch_normalization_26_moving_mean:	�I
:assignvariableop_32_batch_normalization_26_moving_variance:	�E
6assignvariableop_33_batch_normalization_27_moving_mean:	�I
:assignvariableop_34_batch_normalization_27_moving_variance:	�#
assignvariableop_35_total: #
assignvariableop_36_count: %
assignvariableop_37_total_1: %
assignvariableop_38_count_1: =
*assignvariableop_39_adam_dense_20_kernel_m:	�6
(assignvariableop_40_adam_dense_20_bias_m:E
7assignvariableop_41_adam_batch_normalization_24_gamma_m:D
6assignvariableop_42_adam_batch_normalization_24_beta_m:E
+assignvariableop_43_adam_conv2d_24_kernel_m: 7
)assignvariableop_44_adam_conv2d_24_bias_m: E
7assignvariableop_45_adam_batch_normalization_25_gamma_m: D
6assignvariableop_46_adam_batch_normalization_25_beta_m: F
+assignvariableop_47_adam_conv2d_25_kernel_m: �8
)assignvariableop_48_adam_conv2d_25_bias_m:	�F
7assignvariableop_49_adam_batch_normalization_26_gamma_m:	�E
6assignvariableop_50_adam_batch_normalization_26_beta_m:	�G
+assignvariableop_51_adam_conv2d_26_kernel_m:��8
)assignvariableop_52_adam_conv2d_26_bias_m:	�F
7assignvariableop_53_adam_batch_normalization_27_gamma_m:	�E
6assignvariableop_54_adam_batch_normalization_27_beta_m:	�G
+assignvariableop_55_adam_conv2d_27_kernel_m:��8
)assignvariableop_56_adam_conv2d_27_bias_m:	�?
*assignvariableop_57_adam_dense_18_kernel_m:��*�7
(assignvariableop_58_adam_dense_18_bias_m:	�>
*assignvariableop_59_adam_dense_19_kernel_m:
��7
(assignvariableop_60_adam_dense_19_bias_m:	�=
*assignvariableop_61_adam_dense_20_kernel_v:	�6
(assignvariableop_62_adam_dense_20_bias_v:E
7assignvariableop_63_adam_batch_normalization_24_gamma_v:D
6assignvariableop_64_adam_batch_normalization_24_beta_v:E
+assignvariableop_65_adam_conv2d_24_kernel_v: 7
)assignvariableop_66_adam_conv2d_24_bias_v: E
7assignvariableop_67_adam_batch_normalization_25_gamma_v: D
6assignvariableop_68_adam_batch_normalization_25_beta_v: F
+assignvariableop_69_adam_conv2d_25_kernel_v: �8
)assignvariableop_70_adam_conv2d_25_bias_v:	�F
7assignvariableop_71_adam_batch_normalization_26_gamma_v:	�E
6assignvariableop_72_adam_batch_normalization_26_beta_v:	�G
+assignvariableop_73_adam_conv2d_26_kernel_v:��8
)assignvariableop_74_adam_conv2d_26_bias_v:	�F
7assignvariableop_75_adam_batch_normalization_27_gamma_v:	�E
6assignvariableop_76_adam_batch_normalization_27_beta_v:	�G
+assignvariableop_77_adam_conv2d_27_kernel_v:��8
)assignvariableop_78_adam_conv2d_27_bias_v:	�?
*assignvariableop_79_adam_dense_18_kernel_v:��*�7
(assignvariableop_80_adam_dense_18_bias_v:	�>
*assignvariableop_81_adam_dense_19_kernel_v:
��7
(assignvariableop_82_adam_dense_19_bias_v:	�
identity_84��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_9�*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�)
value�)B�)TB)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_24_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_24_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_24_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_24_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_25_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_25_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_25_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_25_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_26_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_26_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_26_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_26_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp0assignvariableop_19_batch_normalization_27_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_27_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp$assignvariableop_21_conv2d_27_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp"assignvariableop_22_conv2d_27_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp#assignvariableop_23_dense_18_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_dense_18_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp#assignvariableop_25_dense_19_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp!assignvariableop_26_dense_19_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_batch_normalization_24_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp:assignvariableop_28_batch_normalization_24_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp6assignvariableop_29_batch_normalization_25_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp:assignvariableop_30_batch_normalization_25_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp6assignvariableop_31_batch_normalization_26_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp:assignvariableop_32_batch_normalization_26_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp6assignvariableop_33_batch_normalization_27_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp:assignvariableop_34_batch_normalization_27_moving_varianceIdentity_34:output:0"/device:CPU:0*
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
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_20_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_20_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_24_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_24_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_24_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_24_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_25_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_25_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_25_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_25_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_26_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_26_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_26_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_26_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_27_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_27_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_27_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_27_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_18_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_18_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_19_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_19_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_20_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_20_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_24_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_24_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_24_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_24_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_25_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_25_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_25_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_25_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_26_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_26_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_26_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_26_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_27_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_27_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_27_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_27_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_18_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_18_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_19_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_19_bias_vIdentity_82:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_3196127

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
�
�
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198801

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3194833

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
 
_user_specified_nameinputs"�L
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
trainable_variables
regularization_losses
	variables
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
trainable_variables
regularization_losses
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"��
_tf_keras_sequential��{"name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_6_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT52AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 49, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_6_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_6_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT52AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 26}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 34}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 38}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 41}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 42}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 43}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 46}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 47}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 48}]}}}
�

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 51}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
"iter

#beta_1

$beta_2
	%decay
&learning_ratem�m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�4m�5m�6m�7m�8m�9m�:m�v�v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�4v�5v�6v�7v�8v�9v�:v�"
	optimizer
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
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
'0
(1
;2
<3
)4
*5
+6
,7
=8
>9
-10
.11
/12
013
?14
@15
116
217
318
419
A20
B21
522
623
724
825
926
:27
28
29"
trackable_list_wrapper
�
trainable_variables

Clayers
Dlayer_metrics
Elayer_regularization_losses
regularization_losses
Fmetrics
	variables
Gnon_trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "lambda_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT52AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
�

Laxis
	'gamma
(beta
;moving_mean
<moving_variance
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

)kernel
*bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 55}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

Uaxis
	+gamma
,beta
=moving_mean
>moving_variance
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 32]}}
�


-kernel
.bias
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 57}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 32]}}
�
^trainable_variables
_regularization_losses
`	variables
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 58}}
�

baxis
	/gamma
0beta
?moving_mean
@moving_variance
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 128]}}
�


1kernel
2bias
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 128]}}
�

kaxis
	3gamma
4beta
Amoving_mean
Bmoving_variance
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}, "shared_object_id": 61}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 256]}}
�


5kernel
6bias
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 256]}}
�
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 63}}
�	

7kernel
8bias
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 41}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 700928}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 700928]}}
�
|trainable_variables
}regularization_losses
~	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 43}
�	

9kernel
:bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 46}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 48}
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
:19"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
�
'0
(1
;2
<3
)4
*5
+6
,7
=8
>9
-10
.11
/12
013
?14
@15
116
217
318
419
A20
B21
522
623
724
825
926
:27"
trackable_list_wrapper
�
trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
regularization_losses
�metrics
	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_20/kernel
:2dense_20/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
�layers
�layer_metrics
regularization_losses
�metrics
�non_trainable_variables
 	variables
 �layer_regularization_losses
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
*:(2batch_normalization_24/gamma
):'2batch_normalization_24/beta
*:( 2conv2d_24/kernel
: 2conv2d_24/bias
*:( 2batch_normalization_25/gamma
):' 2batch_normalization_25/beta
+:) �2conv2d_25/kernel
:�2conv2d_25/bias
+:)�2batch_normalization_26/gamma
*:(�2batch_normalization_26/beta
,:*��2conv2d_26/kernel
:�2conv2d_26/bias
+:)�2batch_normalization_27/gamma
*:(�2batch_normalization_27/beta
,:*��2conv2d_27/kernel
:�2conv2d_27/bias
$:"��*�2dense_18/kernel
:�2dense_18/bias
#:!
��2dense_19/kernel
:�2dense_19/bias
2:0 (2"batch_normalization_24/moving_mean
6:4 (2&batch_normalization_24/moving_variance
2:0  (2"batch_normalization_25/moving_mean
6:4  (2&batch_normalization_25/moving_variance
3:1� (2"batch_normalization_26/moving_mean
7:5� (2&batch_normalization_26/moving_variance
3:1� (2"batch_normalization_27/moving_mean
7:5� (2&batch_normalization_27/moving_variance
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
X
;0
<1
=2
>3
?4
@5
A6
B7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Htrainable_variables
�layers
�layer_metrics
Iregularization_losses
�metrics
�non_trainable_variables
J	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
'0
(1
;2
<3"
trackable_list_wrapper
�
Mtrainable_variables
�layers
�layer_metrics
Nregularization_losses
�metrics
�non_trainable_variables
O	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
Qtrainable_variables
�layers
�layer_metrics
Rregularization_losses
�metrics
�non_trainable_variables
S	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
+0
,1
=2
>3"
trackable_list_wrapper
�
Vtrainable_variables
�layers
�layer_metrics
Wregularization_losses
�metrics
�non_trainable_variables
X	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
Ztrainable_variables
�layers
�layer_metrics
[regularization_losses
�metrics
�non_trainable_variables
\	variables
 �layer_regularization_losses
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
^trainable_variables
�layers
�layer_metrics
_regularization_losses
�metrics
�non_trainable_variables
`	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
<
/0
01
?2
@3"
trackable_list_wrapper
�
ctrainable_variables
�layers
�layer_metrics
dregularization_losses
�metrics
�non_trainable_variables
e	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
gtrainable_variables
�layers
�layer_metrics
hregularization_losses
�metrics
�non_trainable_variables
i	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
<
30
41
A2
B3"
trackable_list_wrapper
�
ltrainable_variables
�layers
�layer_metrics
mregularization_losses
�metrics
�non_trainable_variables
n	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
�
ptrainable_variables
�layers
�layer_metrics
qregularization_losses
�metrics
�non_trainable_variables
r	variables
 �layer_regularization_losses
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
ttrainable_variables
�layers
�layer_metrics
uregularization_losses
�metrics
�non_trainable_variables
v	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
�
xtrainable_variables
�layers
�layer_metrics
yregularization_losses
�metrics
�non_trainable_variables
z	variables
 �layer_regularization_losses
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
|trainable_variables
�layers
�layer_metrics
}regularization_losses
�metrics
�non_trainable_variables
~	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
90
:1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
�
�trainable_variables
�layers
�layer_metrics
�regularization_losses
�metrics
�non_trainable_variables
�	variables
 �layer_regularization_losses
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
�trainable_variables
�layers
�layer_metrics
�regularization_losses
�metrics
�non_trainable_variables
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
X
;0
<1
=2
>3
?4
@5
A6
B7"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
;0
<1"
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
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
=0
>1"
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
 "
trackable_list_wrapper
.
?0
@1"
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
.
A0
B1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
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
(
�0"
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
':%	�2Adam/dense_20/kernel/m
 :2Adam/dense_20/bias/m
/:-2#Adam/batch_normalization_24/gamma/m
.:,2"Adam/batch_normalization_24/beta/m
/:- 2Adam/conv2d_24/kernel/m
!: 2Adam/conv2d_24/bias/m
/:- 2#Adam/batch_normalization_25/gamma/m
.:, 2"Adam/batch_normalization_25/beta/m
0:. �2Adam/conv2d_25/kernel/m
": �2Adam/conv2d_25/bias/m
0:.�2#Adam/batch_normalization_26/gamma/m
/:-�2"Adam/batch_normalization_26/beta/m
1:/��2Adam/conv2d_26/kernel/m
": �2Adam/conv2d_26/bias/m
0:.�2#Adam/batch_normalization_27/gamma/m
/:-�2"Adam/batch_normalization_27/beta/m
1:/��2Adam/conv2d_27/kernel/m
": �2Adam/conv2d_27/bias/m
):'��*�2Adam/dense_18/kernel/m
!:�2Adam/dense_18/bias/m
(:&
��2Adam/dense_19/kernel/m
!:�2Adam/dense_19/bias/m
':%	�2Adam/dense_20/kernel/v
 :2Adam/dense_20/bias/v
/:-2#Adam/batch_normalization_24/gamma/v
.:,2"Adam/batch_normalization_24/beta/v
/:- 2Adam/conv2d_24/kernel/v
!: 2Adam/conv2d_24/bias/v
/:- 2#Adam/batch_normalization_25/gamma/v
.:, 2"Adam/batch_normalization_25/beta/v
0:. �2Adam/conv2d_25/kernel/v
": �2Adam/conv2d_25/bias/v
0:.�2#Adam/batch_normalization_26/gamma/v
/:-�2"Adam/batch_normalization_26/beta/v
1:/��2Adam/conv2d_26/kernel/v
": �2Adam/conv2d_26/bias/v
0:.�2#Adam/batch_normalization_27/gamma/v
/:-�2"Adam/batch_normalization_27/beta/v
1:/��2Adam/conv2d_27/kernel/v
": �2Adam/conv2d_27/bias/v
):'��*�2Adam/dense_18/kernel/v
!:�2Adam/dense_18/bias/v
(:&
��2Adam/dense_19/kernel/v
!:�2Adam/dense_19/bias/v
�2�
A__inference_CNN3_layer_call_and_return_conditional_losses_3196899
A__inference_CNN3_layer_call_and_return_conditional_losses_3197049
A__inference_CNN3_layer_call_and_return_conditional_losses_3197185
A__inference_CNN3_layer_call_and_return_conditional_losses_3197335�
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
"__inference__wrapped_model_3194421�
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
&__inference_CNN3_layer_call_fn_3197400
&__inference_CNN3_layer_call_fn_3197465
&__inference_CNN3_layer_call_fn_3197530
&__inference_CNN3_layer_call_fn_3197595�
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
__inference_call_2942419
__inference_call_2942537
__inference_call_2942655�
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_3197742
I__inference_sequential_6_layer_call_and_return_conditional_losses_3197885
I__inference_sequential_6_layer_call_and_return_conditional_losses_3198014
I__inference_sequential_6_layer_call_and_return_conditional_losses_3198157�
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
.__inference_sequential_6_layer_call_fn_3198218
.__inference_sequential_6_layer_call_fn_3198279
.__inference_sequential_6_layer_call_fn_3198340
.__inference_sequential_6_layer_call_fn_3198401�
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
E__inference_dense_20_layer_call_and_return_conditional_losses_3198412�
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
*__inference_dense_20_layer_call_fn_3198421�
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
%__inference_signature_wrapper_3196763input_1"�
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
E__inference_lambda_6_layer_call_and_return_conditional_losses_3198429
E__inference_lambda_6_layer_call_and_return_conditional_losses_3198437�
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
*__inference_lambda_6_layer_call_fn_3198442
*__inference_lambda_6_layer_call_fn_3198447�
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
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198465
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198483
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198501
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198519�
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
8__inference_batch_normalization_24_layer_call_fn_3198532
8__inference_batch_normalization_24_layer_call_fn_3198545
8__inference_batch_normalization_24_layer_call_fn_3198558
8__inference_batch_normalization_24_layer_call_fn_3198571�
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
F__inference_conv2d_24_layer_call_and_return_conditional_losses_3198594�
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
+__inference_conv2d_24_layer_call_fn_3198603�
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
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198621
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198639
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198657
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198675�
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
8__inference_batch_normalization_25_layer_call_fn_3198688
8__inference_batch_normalization_25_layer_call_fn_3198701
8__inference_batch_normalization_25_layer_call_fn_3198714
8__inference_batch_normalization_25_layer_call_fn_3198727�
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
F__inference_conv2d_25_layer_call_and_return_conditional_losses_3198738�
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
+__inference_conv2d_25_layer_call_fn_3198747�
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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3194679�
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
1__inference_max_pooling2d_6_layer_call_fn_3194685�
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
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198765
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198783
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198801
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198819�
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
8__inference_batch_normalization_26_layer_call_fn_3198832
8__inference_batch_normalization_26_layer_call_fn_3198845
8__inference_batch_normalization_26_layer_call_fn_3198858
8__inference_batch_normalization_26_layer_call_fn_3198871�
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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_3198882�
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
+__inference_conv2d_26_layer_call_fn_3198891�
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198909
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198927
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198945
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198963�
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
8__inference_batch_normalization_27_layer_call_fn_3198976
8__inference_batch_normalization_27_layer_call_fn_3198989
8__inference_batch_normalization_27_layer_call_fn_3199002
8__inference_batch_normalization_27_layer_call_fn_3199015�
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_3199026�
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
+__inference_conv2d_27_layer_call_fn_3199035�
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
F__inference_flatten_6_layer_call_and_return_conditional_losses_3199041�
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
+__inference_flatten_6_layer_call_fn_3199046�
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
E__inference_dense_18_layer_call_and_return_conditional_losses_3199069�
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
*__inference_dense_18_layer_call_fn_3199078�
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
G__inference_dropout_12_layer_call_and_return_conditional_losses_3199083
G__inference_dropout_12_layer_call_and_return_conditional_losses_3199095�
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
,__inference_dropout_12_layer_call_fn_3199100
,__inference_dropout_12_layer_call_fn_3199105�
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
E__inference_dense_19_layer_call_and_return_conditional_losses_3199128�
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
*__inference_dense_19_layer_call_fn_3199137�
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
G__inference_dropout_13_layer_call_and_return_conditional_losses_3199142
G__inference_dropout_13_layer_call_and_return_conditional_losses_3199154�
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
,__inference_dropout_13_layer_call_fn_3199159
,__inference_dropout_13_layer_call_fn_3199164�
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
__inference_loss_fn_0_3199175�
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
__inference_loss_fn_1_3199186�
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
__inference_loss_fn_2_3199197�
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
A__inference_CNN3_layer_call_and_return_conditional_losses_3196899�'(;<)*+,=>-./0?@1234AB56789:;�8
1�.
(�%
inputs���������KK
p 
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_3197049�'(;<)*+,=>-./0?@1234AB56789:;�8
1�.
(�%
inputs���������KK
p
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_3197185�'(;<)*+,=>-./0?@1234AB56789:<�9
2�/
)�&
input_1���������KK
p 
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_3197335�'(;<)*+,=>-./0?@1234AB56789:<�9
2�/
)�&
input_1���������KK
p
� "%�"
�
0���������
� �
&__inference_CNN3_layer_call_fn_3197400x'(;<)*+,=>-./0?@1234AB56789:<�9
2�/
)�&
input_1���������KK
p 
� "�����������
&__inference_CNN3_layer_call_fn_3197465w'(;<)*+,=>-./0?@1234AB56789:;�8
1�.
(�%
inputs���������KK
p 
� "�����������
&__inference_CNN3_layer_call_fn_3197530w'(;<)*+,=>-./0?@1234AB56789:;�8
1�.
(�%
inputs���������KK
p
� "�����������
&__inference_CNN3_layer_call_fn_3197595x'(;<)*+,=>-./0?@1234AB56789:<�9
2�/
)�&
input_1���������KK
p
� "�����������
"__inference__wrapped_model_3194421�'(;<)*+,=>-./0?@1234AB56789:8�5
.�+
)�&
input_1���������KK
� "3�0
.
output_1"�
output_1����������
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198465�'(;<M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198483�'(;<M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198501r'(;<;�8
1�.
(�%
inputs���������KK
p 
� "-�*
#� 
0���������KK
� �
S__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3198519r'(;<;�8
1�.
(�%
inputs���������KK
p
� "-�*
#� 
0���������KK
� �
8__inference_batch_normalization_24_layer_call_fn_3198532�'(;<M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
8__inference_batch_normalization_24_layer_call_fn_3198545�'(;<M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
8__inference_batch_normalization_24_layer_call_fn_3198558e'(;<;�8
1�.
(�%
inputs���������KK
p 
� " ����������KK�
8__inference_batch_normalization_24_layer_call_fn_3198571e'(;<;�8
1�.
(�%
inputs���������KK
p
� " ����������KK�
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198621�+,=>M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198639�+,=>M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198657r+,=>;�8
1�.
(�%
inputs���������KK 
p 
� "-�*
#� 
0���������KK 
� �
S__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3198675r+,=>;�8
1�.
(�%
inputs���������KK 
p
� "-�*
#� 
0���������KK 
� �
8__inference_batch_normalization_25_layer_call_fn_3198688�+,=>M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
8__inference_batch_normalization_25_layer_call_fn_3198701�+,=>M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
8__inference_batch_normalization_25_layer_call_fn_3198714e+,=>;�8
1�.
(�%
inputs���������KK 
p 
� " ����������KK �
8__inference_batch_normalization_25_layer_call_fn_3198727e+,=>;�8
1�.
(�%
inputs���������KK 
p
� " ����������KK �
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198765�/0?@N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198783�/0?@N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198801t/0?@<�9
2�/
)�&
inputs���������%%�
p 
� ".�+
$�!
0���������%%�
� �
S__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3198819t/0?@<�9
2�/
)�&
inputs���������%%�
p
� ".�+
$�!
0���������%%�
� �
8__inference_batch_normalization_26_layer_call_fn_3198832�/0?@N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_26_layer_call_fn_3198845�/0?@N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_26_layer_call_fn_3198858g/0?@<�9
2�/
)�&
inputs���������%%�
p 
� "!����������%%��
8__inference_batch_normalization_26_layer_call_fn_3198871g/0?@<�9
2�/
)�&
inputs���������%%�
p
� "!����������%%��
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198909�34ABN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198927�34ABN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198945t34AB<�9
2�/
)�&
inputs���������%%�
p 
� ".�+
$�!
0���������%%�
� �
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3198963t34AB<�9
2�/
)�&
inputs���������%%�
p
� ".�+
$�!
0���������%%�
� �
8__inference_batch_normalization_27_layer_call_fn_3198976�34ABN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_27_layer_call_fn_3198989�34ABN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_27_layer_call_fn_3199002g34AB<�9
2�/
)�&
inputs���������%%�
p 
� "!����������%%��
8__inference_batch_normalization_27_layer_call_fn_3199015g34AB<�9
2�/
)�&
inputs���������%%�
p
� "!����������%%��
__inference_call_2942419g'(;<)*+,=>-./0?@1234AB56789:3�0
)�&
 �
inputs�KK
p
� "�	��
__inference_call_2942537g'(;<)*+,=>-./0?@1234AB56789:3�0
)�&
 �
inputs�KK
p 
� "�	��
__inference_call_2942655w'(;<)*+,=>-./0?@1234AB56789:;�8
1�.
(�%
inputs���������KK
p 
� "�����������
F__inference_conv2d_24_layer_call_and_return_conditional_losses_3198594l)*7�4
-�*
(�%
inputs���������KK
� "-�*
#� 
0���������KK 
� �
+__inference_conv2d_24_layer_call_fn_3198603_)*7�4
-�*
(�%
inputs���������KK
� " ����������KK �
F__inference_conv2d_25_layer_call_and_return_conditional_losses_3198738m-.7�4
-�*
(�%
inputs���������KK 
� ".�+
$�!
0���������KK�
� �
+__inference_conv2d_25_layer_call_fn_3198747`-.7�4
-�*
(�%
inputs���������KK 
� "!����������KK��
F__inference_conv2d_26_layer_call_and_return_conditional_losses_3198882n128�5
.�+
)�&
inputs���������%%�
� ".�+
$�!
0���������%%�
� �
+__inference_conv2d_26_layer_call_fn_3198891a128�5
.�+
)�&
inputs���������%%�
� "!����������%%��
F__inference_conv2d_27_layer_call_and_return_conditional_losses_3199026n568�5
.�+
)�&
inputs���������%%�
� ".�+
$�!
0���������%%�
� �
+__inference_conv2d_27_layer_call_fn_3199035a568�5
.�+
)�&
inputs���������%%�
� "!����������%%��
E__inference_dense_18_layer_call_and_return_conditional_losses_3199069_781�.
'�$
"�
inputs�����������*
� "&�#
�
0����������
� �
*__inference_dense_18_layer_call_fn_3199078R781�.
'�$
"�
inputs�����������*
� "������������
E__inference_dense_19_layer_call_and_return_conditional_losses_3199128^9:0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_19_layer_call_fn_3199137Q9:0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_20_layer_call_and_return_conditional_losses_3198412]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_20_layer_call_fn_3198421P0�-
&�#
!�
inputs����������
� "�����������
G__inference_dropout_12_layer_call_and_return_conditional_losses_3199083^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_12_layer_call_and_return_conditional_losses_3199095^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_12_layer_call_fn_3199100Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_12_layer_call_fn_3199105Q4�1
*�'
!�
inputs����������
p
� "������������
G__inference_dropout_13_layer_call_and_return_conditional_losses_3199142^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_13_layer_call_and_return_conditional_losses_3199154^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_13_layer_call_fn_3199159Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_13_layer_call_fn_3199164Q4�1
*�'
!�
inputs����������
p
� "������������
F__inference_flatten_6_layer_call_and_return_conditional_losses_3199041c8�5
.�+
)�&
inputs���������%%�
� "'�$
�
0�����������*
� �
+__inference_flatten_6_layer_call_fn_3199046V8�5
.�+
)�&
inputs���������%%�
� "������������*�
E__inference_lambda_6_layer_call_and_return_conditional_losses_3198429p?�<
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
E__inference_lambda_6_layer_call_and_return_conditional_losses_3198437p?�<
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
*__inference_lambda_6_layer_call_fn_3198442c?�<
5�2
(�%
inputs���������KK

 
p 
� " ����������KK�
*__inference_lambda_6_layer_call_fn_3198447c?�<
5�2
(�%
inputs���������KK

 
p
� " ����������KK<
__inference_loss_fn_0_3199175)�

� 
� "� <
__inference_loss_fn_1_31991867�

� 
� "� <
__inference_loss_fn_2_31991979�

� 
� "� �
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3194679�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_6_layer_call_fn_3194685�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_6_layer_call_and_return_conditional_losses_3197742�'(;<)*+,=>-./0?@1234AB56789:?�<
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_3197885�'(;<)*+,=>-./0?@1234AB56789:?�<
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_3198014�'(;<)*+,=>-./0?@1234AB56789:G�D
=�:
0�-
lambda_6_input���������KK
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_6_layer_call_and_return_conditional_losses_3198157�'(;<)*+,=>-./0?@1234AB56789:G�D
=�:
0�-
lambda_6_input���������KK
p

 
� "&�#
�
0����������
� �
.__inference_sequential_6_layer_call_fn_3198218�'(;<)*+,=>-./0?@1234AB56789:G�D
=�:
0�-
lambda_6_input���������KK
p 

 
� "������������
.__inference_sequential_6_layer_call_fn_3198279z'(;<)*+,=>-./0?@1234AB56789:?�<
5�2
(�%
inputs���������KK
p 

 
� "������������
.__inference_sequential_6_layer_call_fn_3198340z'(;<)*+,=>-./0?@1234AB56789:?�<
5�2
(�%
inputs���������KK
p

 
� "������������
.__inference_sequential_6_layer_call_fn_3198401�'(;<)*+,=>-./0?@1234AB56789:G�D
=�:
0�-
lambda_6_input���������KK
p

 
� "������������
%__inference_signature_wrapper_3196763�'(;<)*+,=>-./0?@1234AB56789:C�@
� 
9�6
4
input_1)�&
input_1���������KK"3�0
.
output_1"�
output_1���������