��%
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
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ݿ 
{
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_39/kernel
t
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes
:	�*
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
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
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_9/gamma
�
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_9/beta
�
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_9/moving_mean
�
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_9/moving_variance
�
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:*
dtype0
�
conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_36/kernel
}
$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*&
_output_shapes
: *
dtype0
t
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_36/bias
m
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes
: *
dtype0
�
conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*!
shared_nameconv2d_37/kernel
~
$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*'
_output_shapes
: �*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:�*
dtype0
�
conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_38/kernel

$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_38/bias
n
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes	
:�*
dtype0
�
conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_39/kernel

$conv2d_39/kernel/Read/ReadVariableOpReadVariableOpconv2d_39/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_39/bias
n
"conv2d_39/bias/Read/ReadVariableOpReadVariableOpconv2d_39/bias*
_output_shapes	
:�*
dtype0
|
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�@�* 
shared_namedense_36/kernel
u
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel* 
_output_shapes
:
�@�*
dtype0
s
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_36/bias
l
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes	
:�*
dtype0
|
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_37/kernel
u
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel* 
_output_shapes
:
��*
dtype0
s
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_37/bias
l
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes	
:�*
dtype0
|
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_38/kernel
u
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel* 
_output_shapes
:
��*
dtype0
s
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_38/bias
l
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
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
Adam/dense_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_39/kernel/m
�
*Adam/dense_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_39/bias/m
y
(Adam/dense_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/m*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_9/gamma/m
�
6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/m*
_output_shapes
:*
dtype0
�
!Adam/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_9/beta/m
�
5Adam/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_36/kernel/m
�
+Adam/conv2d_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_36/bias/m
{
)Adam/conv2d_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_37/kernel/m
�
+Adam/conv2d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/m*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_37/bias/m
|
)Adam/conv2d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_38/kernel/m
�
+Adam/conv2d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_38/bias/m
|
)Adam/conv2d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_39/kernel/m
�
+Adam/conv2d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_39/bias/m
|
)Adam/conv2d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�@�*'
shared_nameAdam/dense_36/kernel/m
�
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m* 
_output_shapes
:
�@�*
dtype0
�
Adam/dense_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_36/bias/m
z
(Adam/dense_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_37/kernel/m
�
*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_37/bias/m
z
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_38/kernel/m
�
*Adam/dense_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_38/bias/m
z
(Adam/dense_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_39/kernel/v
�
*Adam/dense_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_39/bias/v
y
(Adam/dense_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/v*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_9/gamma/v
�
6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/v*
_output_shapes
:*
dtype0
�
!Adam/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_9/beta/v
�
5Adam/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_36/kernel/v
�
+Adam/conv2d_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_36/bias/v
{
)Adam/conv2d_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_37/kernel/v
�
+Adam/conv2d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/v*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_37/bias/v
|
)Adam/conv2d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_38/kernel/v
�
+Adam/conv2d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_38/bias/v
|
)Adam/conv2d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_39/kernel/v
�
+Adam/conv2d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_39/bias/v
|
)Adam/conv2d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�@�*'
shared_nameAdam/dense_36/kernel/v
�
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v* 
_output_shapes
:
�@�*
dtype0
�
Adam/dense_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_36/bias/v
z
(Adam/dense_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_37/kernel/v
�
*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_37/bias/v
z
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_38/kernel/v
�
*Adam/dense_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_38/bias/v
z
(Adam/dense_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�s
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�s
value�sB�s B�s
�

h2ptjl
_output
	optimizer
regularization_losses
	variables
trainable_variables
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
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer_with_weights-4
layer-8
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
�
%iter

&beta_1

'beta_2
	(decay
)learning_ratem� m�*m�+m�.m�/m�0m�1m�2m�3m�4m�5m�6m�7m�8m�9m�:m�;m�v� v�*v�+v�.v�/v�0v�1v�2v�3v�4v�5v�6v�7v�8v�9v�:v�;v�
 
�
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17
18
 19
�
*0
+1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
16
 17
�

<layers
regularization_losses
	variables
=metrics
trainable_variables
>non_trainable_variables
?layer_regularization_losses
@layer_metrics
 
R
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
�
Eaxis
	*gamma
+beta
,moving_mean
-moving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

.kernel
/bias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
h

0kernel
1bias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
R
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
h

2kernel
3bias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
R
^regularization_losses
_	variables
`trainable_variables
a	keras_api
h

4kernel
5bias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
R
fregularization_losses
g	variables
htrainable_variables
i	keras_api
R
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
R
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
h

6kernel
7bias
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
R
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
h

8kernel
9bias
zregularization_losses
{	variables
|trainable_variables
}	keras_api
T
~regularization_losses
	variables
�trainable_variables
�	keras_api
l

:kernel
;bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 
�
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17
v
*0
+1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
�
�layers
regularization_losses
	variables
�metrics
trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
NL
VARIABLE_VALUEdense_39/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_39/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
�
�layers
!regularization_losses
"	variables
�metrics
#trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
WU
VARIABLE_VALUEbatch_normalization_9/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_9/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_9/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_9/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_36/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_36/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_37/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_37/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_38/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_38/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_39/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_39/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_36/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_36/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_37/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_37/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_38/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_38/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE

0
1

�0
�1

,0
-1
 
 
 
 
 
�
�layers
Aregularization_losses
B	variables
�metrics
Ctrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 

*0
+1
,2
-3

*0
+1
�
�layers
Fregularization_losses
G	variables
�metrics
Htrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 

.0
/1

.0
/1
�
�layers
Jregularization_losses
K	variables
�metrics
Ltrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
Nregularization_losses
O	variables
�metrics
Ptrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 

00
11

00
11
�
�layers
Rregularization_losses
S	variables
�metrics
Ttrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
Vregularization_losses
W	variables
�metrics
Xtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 

20
31

20
31
�
�layers
Zregularization_losses
[	variables
�metrics
\trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
^regularization_losses
_	variables
�metrics
`trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 

40
51

40
51
�
�layers
bregularization_losses
c	variables
�metrics
dtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
fregularization_losses
g	variables
�metrics
htrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
jregularization_losses
k	variables
�metrics
ltrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
nregularization_losses
o	variables
�metrics
ptrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 

60
71

60
71
�
�layers
rregularization_losses
s	variables
�metrics
ttrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
vregularization_losses
w	variables
�metrics
xtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 

80
91

80
91
�
�layers
zregularization_losses
{	variables
�metrics
|trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
~regularization_losses
	variables
�metrics
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 

:0
;1

:0
;1
�
�layers
�regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
 
 
 
�
�layers
�regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
14
15
16
17
 

,0
-1
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
 
 

,0
-1
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
VARIABLE_VALUEAdam/dense_39/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_39/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_9/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_36/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_36/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_37/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_37/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_38/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_38/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_39/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_39/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_36/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_36/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_37/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_37/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_38/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_38/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_39/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_39/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_9/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_36/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_36/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_37/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_37/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_38/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_38/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_39/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_39/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_36/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_36/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_37/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_37/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_38/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_38/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������KK*
dtype0*$
shape:���������KK
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasdense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/biasdense_39/kerneldense_39/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_1727973
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOp#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_39/kernel/m/Read/ReadVariableOp(Adam/dense_39/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp+Adam/conv2d_36/kernel/m/Read/ReadVariableOp)Adam/conv2d_36/bias/m/Read/ReadVariableOp+Adam/conv2d_37/kernel/m/Read/ReadVariableOp)Adam/conv2d_37/bias/m/Read/ReadVariableOp+Adam/conv2d_38/kernel/m/Read/ReadVariableOp)Adam/conv2d_38/bias/m/Read/ReadVariableOp+Adam/conv2d_39/kernel/m/Read/ReadVariableOp)Adam/conv2d_39/bias/m/Read/ReadVariableOp*Adam/dense_36/kernel/m/Read/ReadVariableOp(Adam/dense_36/bias/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp*Adam/dense_38/kernel/m/Read/ReadVariableOp(Adam/dense_38/bias/m/Read/ReadVariableOp*Adam/dense_39/kernel/v/Read/ReadVariableOp(Adam/dense_39/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp+Adam/conv2d_36/kernel/v/Read/ReadVariableOp)Adam/conv2d_36/bias/v/Read/ReadVariableOp+Adam/conv2d_37/kernel/v/Read/ReadVariableOp)Adam/conv2d_37/bias/v/Read/ReadVariableOp+Adam/conv2d_38/kernel/v/Read/ReadVariableOp)Adam/conv2d_38/bias/v/Read/ReadVariableOp+Adam/conv2d_39/kernel/v/Read/ReadVariableOp)Adam/conv2d_39/bias/v/Read/ReadVariableOp*Adam/dense_36/kernel/v/Read/ReadVariableOp(Adam/dense_36/bias/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOp*Adam/dense_38/kernel/v/Read/ReadVariableOp(Adam/dense_38/bias/v/Read/ReadVariableOpConst*N
TinG
E2C	*
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
GPU2 *0J 8� *)
f$R"
 __inference__traced_save_1730060
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_39/kerneldense_39/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasdense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/biastotalcounttotal_1count_1Adam/dense_39/kernel/mAdam/dense_39/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/conv2d_36/kernel/mAdam/conv2d_36/bias/mAdam/conv2d_37/kernel/mAdam/conv2d_37/bias/mAdam/conv2d_38/kernel/mAdam/conv2d_38/bias/mAdam/conv2d_39/kernel/mAdam/conv2d_39/bias/mAdam/dense_36/kernel/mAdam/dense_36/bias/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/dense_38/kernel/mAdam/dense_38/bias/mAdam/dense_39/kernel/vAdam/dense_39/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/conv2d_36/kernel/vAdam/conv2d_36/bias/vAdam/conv2d_37/kernel/vAdam/conv2d_37/bias/vAdam/conv2d_38/kernel/vAdam/conv2d_38/bias/vAdam/conv2d_39/kernel/vAdam/conv2d_39/bias/vAdam/dense_36/kernel/vAdam/dense_36/bias/vAdam/dense_37/kernel/vAdam/dense_37/bias/vAdam/dense_38/kernel/vAdam/dense_38/bias/v*M
TinF
D2B*
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
GPU2 *0J 8� *,
f'R%
#__inference__traced_restore_1730265қ
Ϗ
�
__inference_call_1620645

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_38_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_39_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	�H
4sequential_9_dense_36_matmul_readvariableop_resource:
�@�D
5sequential_9_dense_36_biasadd_readvariableop_resource:	�H
4sequential_9_dense_37_matmul_readvariableop_resource:
��D
5sequential_9_dense_37_biasadd_readvariableop_resource:	�H
4sequential_9_dense_38_matmul_readvariableop_resource:
��D
5sequential_9_dense_38_biasadd_readvariableop_resource:	�:
'dense_39_matmul_readvariableop_resource:	�6
(dense_39_biasadd_readvariableop_resource:
identity��dense_39/BiasAdd/ReadVariableOp�dense_39/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�,sequential_9/conv2d_36/Conv2D/ReadVariableOp�-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�,sequential_9/conv2d_37/Conv2D/ReadVariableOp�-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�,sequential_9/conv2d_38/Conv2D/ReadVariableOp�-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�,sequential_9/conv2d_39/Conv2D/ReadVariableOp�,sequential_9/dense_36/BiasAdd/ReadVariableOp�+sequential_9/dense_36/MatMul/ReadVariableOp�,sequential_9/dense_37/BiasAdd/ReadVariableOp�+sequential_9/dense_37/MatMul/ReadVariableOp�,sequential_9/dense_38/BiasAdd/ReadVariableOp�+sequential_9/dense_38/MatMul/ReadVariableOp�
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack�
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1�
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2�
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice�
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp�
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3�
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOp�
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D�
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_9/conv2d_36/BiasAdd�
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_9/conv2d_36/Relu�
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool�
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOp�
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D�
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_9/conv2d_37/BiasAdd�
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_9/conv2d_37/Relu�
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool�
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOp�
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D�
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_9/conv2d_38/BiasAdd�
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_9/conv2d_38/Relu�
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool�
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOp�
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�		�*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D�
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�		�2 
sequential_9/conv2d_39/BiasAdd�
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*(
_output_shapes
:�		�2
sequential_9/conv2d_39/Relu�
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool�
 sequential_9/dropout_36/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*(
_output_shapes
:��2"
 sequential_9/dropout_36/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_36/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0* 
_output_shapes
:
��@2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp�
sequential_9/dense_36/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_36/MatMul�
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp�
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_36/BiasAdd�
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_36/Relu�
 sequential_9/dropout_37/IdentityIdentity(sequential_9/dense_36/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_37/Identity�
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp�
sequential_9/dense_37/MatMulMatMul)sequential_9/dropout_37/Identity:output:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_37/MatMul�
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp�
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_37/BiasAdd�
sequential_9/dense_37/ReluRelu&sequential_9/dense_37/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_37/Relu�
 sequential_9/dropout_38/IdentityIdentity(sequential_9/dense_37/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_38/Identity�
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp�
sequential_9/dense_38/MatMulMatMul)sequential_9/dropout_38/Identity:output:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_38/MatMul�
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp�
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_38/BiasAdd�
sequential_9/dense_38/ReluRelu&sequential_9/dense_38/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_38/Relu�
 sequential_9/dropout_39/IdentityIdentity(sequential_9/dense_38/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_39/Identity�
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_39/MatMul/ReadVariableOp�
dense_39/MatMulMatMul)sequential_9/dropout_39/Identity:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_39/MatMul�
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp�
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_39/BiasAddt
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_39/Softmax�
IdentityIdentitydense_39/Softmax:softmax:0 ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:�KK: : : : : : : : : : : : : : : : : : : : 2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp-sequential_9/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_36/Conv2D/ReadVariableOp,sequential_9/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp-sequential_9/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_37/Conv2D/ReadVariableOp,sequential_9/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp-sequential_9/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_38/Conv2D/ReadVariableOp,sequential_9/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp-sequential_9/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_39/Conv2D/ReadVariableOp,sequential_9/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
�
+__inference_conv2d_38_layer_call_fn_1729563

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
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_17266182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_39_layer_call_fn_1729793

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_17267462
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
��
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1728786

inputs;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_36_conv2d_readvariableop_resource: 7
)conv2d_36_biasadd_readvariableop_resource: C
(conv2d_37_conv2d_readvariableop_resource: �8
)conv2d_37_biasadd_readvariableop_resource:	�D
(conv2d_38_conv2d_readvariableop_resource:��8
)conv2d_38_biasadd_readvariableop_resource:	�D
(conv2d_39_conv2d_readvariableop_resource:��8
)conv2d_39_biasadd_readvariableop_resource:	�;
'dense_36_matmul_readvariableop_resource:
�@�7
(dense_36_biasadd_readvariableop_resource:	�;
'dense_37_matmul_readvariableop_resource:
��7
(dense_37_biasadd_readvariableop_resource:	�;
'dense_38_matmul_readvariableop_resource:
��7
(dense_38_biasadd_readvariableop_resource:	�
identity��5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1� conv2d_36/BiasAdd/ReadVariableOp�conv2d_36/Conv2D/ReadVariableOp�2conv2d_36/kernel/Regularizer/Square/ReadVariableOp� conv2d_37/BiasAdd/ReadVariableOp�conv2d_37/Conv2D/ReadVariableOp� conv2d_38/BiasAdd/ReadVariableOp�conv2d_38/Conv2D/ReadVariableOp� conv2d_39/BiasAdd/ReadVariableOp�conv2d_39/Conv2D/ReadVariableOp�dense_36/BiasAdd/ReadVariableOp�dense_36/MatMul/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�dense_37/BiasAdd/ReadVariableOp�dense_37/MatMul/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�dense_38/BiasAdd/ReadVariableOp�dense_38/MatMul/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�
lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_9/strided_slice/stack�
lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_9/strided_slice/stack_1�
lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_9/strided_slice/stack_2�
lambda_9/strided_sliceStridedSliceinputs%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_9/strided_slice�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOp�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3�
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOp�
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_36/Conv2D�
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp�
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_36/Relu�
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool�
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_37/Conv2D/ReadVariableOp�
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_37/Conv2D�
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp�
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_37/Relu�
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool�
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_38/Conv2D/ReadVariableOp�
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_38/Conv2D�
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp�
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_38/Relu�
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool�
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_39/Conv2D/ReadVariableOp�
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
conv2d_39/Conv2D�
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp�
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
conv2d_39/Relu�
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPool�
dropout_36/IdentityIdentity!max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:����������2
dropout_36/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_36/Identity:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2
flatten_9/Reshape�
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02 
dense_36/MatMul/ReadVariableOp�
dense_36/MatMulMatMulflatten_9/Reshape:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_36/MatMul�
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_36/BiasAdd/ReadVariableOp�
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_36/Relu�
dropout_37/IdentityIdentitydense_36/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_37/Identity�
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_37/MatMul/ReadVariableOp�
dense_37/MatMulMatMuldropout_37/Identity:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_37/MatMul�
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_37/BiasAdd/ReadVariableOp�
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_37/BiasAddt
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_37/Relu�
dropout_38/IdentityIdentitydense_37/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_38/Identity�
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_38/MatMul/ReadVariableOp�
dense_38/MatMulMatMuldropout_38/Identity:output:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_38/MatMul�
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_38/BiasAdd/ReadVariableOp�
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_38/BiasAddt
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_38/Relu�
dropout_39/IdentityIdentitydense_38/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_39/Identity�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentitydropout_39/Identity:output:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
��
�
A__inference_CNN3_layer_call_and_return_conditional_losses_1728225

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_38_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_39_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	�H
4sequential_9_dense_36_matmul_readvariableop_resource:
�@�D
5sequential_9_dense_36_biasadd_readvariableop_resource:	�H
4sequential_9_dense_37_matmul_readvariableop_resource:
��D
5sequential_9_dense_37_biasadd_readvariableop_resource:	�H
4sequential_9_dense_38_matmul_readvariableop_resource:
��D
5sequential_9_dense_38_biasadd_readvariableop_resource:	�:
'dense_39_matmul_readvariableop_resource:	�6
(dense_39_biasadd_readvariableop_resource:
identity��2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�dense_39/BiasAdd/ReadVariableOp�dense_39/MatMul/ReadVariableOp�1sequential_9/batch_normalization_9/AssignNewValue�3sequential_9/batch_normalization_9/AssignNewValue_1�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�,sequential_9/conv2d_36/Conv2D/ReadVariableOp�-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�,sequential_9/conv2d_37/Conv2D/ReadVariableOp�-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�,sequential_9/conv2d_38/Conv2D/ReadVariableOp�-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�,sequential_9/conv2d_39/Conv2D/ReadVariableOp�,sequential_9/dense_36/BiasAdd/ReadVariableOp�+sequential_9/dense_36/MatMul/ReadVariableOp�,sequential_9/dense_37/BiasAdd/ReadVariableOp�+sequential_9/dense_37/MatMul/ReadVariableOp�,sequential_9/dense_38/BiasAdd/ReadVariableOp�+sequential_9/dense_38/MatMul/ReadVariableOp�
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack�
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1�
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2�
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice�
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp�
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<25
3sequential_9/batch_normalization_9/FusedBatchNormV3�
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_9/batch_normalization_9/AssignNewValue�
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_9/batch_normalization_9/AssignNewValue_1�
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOp�
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D�
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_36/BiasAdd�
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_36/Relu�
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool�
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOp�
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D�
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_37/BiasAdd�
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_37/Relu�
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool�
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOp�
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D�
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_38/BiasAdd�
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_38/Relu�
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool�
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOp�
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D�
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2 
sequential_9/conv2d_39/BiasAdd�
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
sequential_9/conv2d_39/Relu�
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool�
%sequential_9/dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2'
%sequential_9/dropout_36/dropout/Const�
#sequential_9/dropout_36/dropout/MulMul.sequential_9/max_pooling2d_39/MaxPool:output:0.sequential_9/dropout_36/dropout/Const:output:0*
T0*0
_output_shapes
:����������2%
#sequential_9/dropout_36/dropout/Mul�
%sequential_9/dropout_36/dropout/ShapeShape.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_36/dropout/Shape�
<sequential_9/dropout_36/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_36/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_36/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=20
.sequential_9/dropout_36/dropout/GreaterEqual/y�
,sequential_9/dropout_36/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_36/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_36/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2.
,sequential_9/dropout_36/dropout/GreaterEqual�
$sequential_9/dropout_36/dropout/CastCast0sequential_9/dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2&
$sequential_9/dropout_36/dropout/Cast�
%sequential_9/dropout_36/dropout/Mul_1Mul'sequential_9/dropout_36/dropout/Mul:z:0(sequential_9/dropout_36/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2'
%sequential_9/dropout_36/dropout/Mul_1�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_36/dropout/Mul_1:z:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp�
sequential_9/dense_36/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/MatMul�
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp�
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/BiasAdd�
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/Relu�
%sequential_9/dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_37/dropout/Const�
#sequential_9/dropout_37/dropout/MulMul(sequential_9/dense_36/Relu:activations:0.sequential_9/dropout_37/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_37/dropout/Mul�
%sequential_9/dropout_37/dropout/ShapeShape(sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_37/dropout/Shape�
<sequential_9/dropout_37/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_37/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_37/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_37/dropout/GreaterEqual/y�
,sequential_9/dropout_37/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_37/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_37/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_37/dropout/GreaterEqual�
$sequential_9/dropout_37/dropout/CastCast0sequential_9/dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_37/dropout/Cast�
%sequential_9/dropout_37/dropout/Mul_1Mul'sequential_9/dropout_37/dropout/Mul:z:0(sequential_9/dropout_37/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_37/dropout/Mul_1�
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp�
sequential_9/dense_37/MatMulMatMul)sequential_9/dropout_37/dropout/Mul_1:z:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/MatMul�
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp�
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/BiasAdd�
sequential_9/dense_37/ReluRelu&sequential_9/dense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/Relu�
%sequential_9/dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_38/dropout/Const�
#sequential_9/dropout_38/dropout/MulMul(sequential_9/dense_37/Relu:activations:0.sequential_9/dropout_38/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_38/dropout/Mul�
%sequential_9/dropout_38/dropout/ShapeShape(sequential_9/dense_37/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_38/dropout/Shape�
<sequential_9/dropout_38/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_38/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_38/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_38/dropout/GreaterEqual/y�
,sequential_9/dropout_38/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_38/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_38/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_38/dropout/GreaterEqual�
$sequential_9/dropout_38/dropout/CastCast0sequential_9/dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_38/dropout/Cast�
%sequential_9/dropout_38/dropout/Mul_1Mul'sequential_9/dropout_38/dropout/Mul:z:0(sequential_9/dropout_38/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_38/dropout/Mul_1�
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp�
sequential_9/dense_38/MatMulMatMul)sequential_9/dropout_38/dropout/Mul_1:z:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/MatMul�
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp�
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/BiasAdd�
sequential_9/dense_38/ReluRelu&sequential_9/dense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/Relu�
%sequential_9/dropout_39/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_39/dropout/Const�
#sequential_9/dropout_39/dropout/MulMul(sequential_9/dense_38/Relu:activations:0.sequential_9/dropout_39/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_39/dropout/Mul�
%sequential_9/dropout_39/dropout/ShapeShape(sequential_9/dense_38/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_39/dropout/Shape�
<sequential_9/dropout_39/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_39/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_39/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_39/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_39/dropout/GreaterEqual/y�
,sequential_9/dropout_39/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_39/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_39/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_39/dropout/GreaterEqual�
$sequential_9/dropout_39/dropout/CastCast0sequential_9/dropout_39/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_39/dropout/Cast�
%sequential_9/dropout_39/dropout/Mul_1Mul'sequential_9/dropout_39/dropout/Mul:z:0(sequential_9/dropout_39/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_39/dropout/Mul_1�
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_39/MatMul/ReadVariableOp�
dense_39/MatMulMatMul)sequential_9/dropout_39/dropout/Mul_1:z:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/MatMul�
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp�
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_39/Softmax�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�

IdentityIdentitydense_39/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_12�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp-sequential_9/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_36/Conv2D/ReadVariableOp,sequential_9/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp-sequential_9/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_37/Conv2D/ReadVariableOp,sequential_9/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp-sequential_9/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_38/Conv2D/ReadVariableOp,sequential_9/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp-sequential_9/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_39/Conv2D/ReadVariableOp,sequential_9/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
*__inference_dense_37_layer_call_fn_1729712

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
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_37_layer_call_and_return_conditional_losses_17267052
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
�
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1729349

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
��
�
__inference_call_1618116

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_38_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_39_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	�H
4sequential_9_dense_36_matmul_readvariableop_resource:
�@�D
5sequential_9_dense_36_biasadd_readvariableop_resource:	�H
4sequential_9_dense_37_matmul_readvariableop_resource:
��D
5sequential_9_dense_37_biasadd_readvariableop_resource:	�H
4sequential_9_dense_38_matmul_readvariableop_resource:
��D
5sequential_9_dense_38_biasadd_readvariableop_resource:	�:
'dense_39_matmul_readvariableop_resource:	�6
(dense_39_biasadd_readvariableop_resource:
identity��dense_39/BiasAdd/ReadVariableOp�dense_39/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�,sequential_9/conv2d_36/Conv2D/ReadVariableOp�-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�,sequential_9/conv2d_37/Conv2D/ReadVariableOp�-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�,sequential_9/conv2d_38/Conv2D/ReadVariableOp�-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�,sequential_9/conv2d_39/Conv2D/ReadVariableOp�,sequential_9/dense_36/BiasAdd/ReadVariableOp�+sequential_9/dense_36/MatMul/ReadVariableOp�,sequential_9/dense_37/BiasAdd/ReadVariableOp�+sequential_9/dense_37/MatMul/ReadVariableOp�,sequential_9/dense_38/BiasAdd/ReadVariableOp�+sequential_9/dense_38/MatMul/ReadVariableOp�
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack�
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1�
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2�
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice�
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp�
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3�
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOp�
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D�
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_36/BiasAdd�
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_36/Relu�
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool�
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOp�
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D�
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_37/BiasAdd�
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_37/Relu�
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool�
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOp�
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D�
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_38/BiasAdd�
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_38/Relu�
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool�
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOp�
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D�
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2 
sequential_9/conv2d_39/BiasAdd�
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
sequential_9/conv2d_39/Relu�
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool�
 sequential_9/dropout_36/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:����������2"
 sequential_9/dropout_36/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_36/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp�
sequential_9/dense_36/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/MatMul�
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp�
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/BiasAdd�
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/Relu�
 sequential_9/dropout_37/IdentityIdentity(sequential_9/dense_36/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_37/Identity�
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp�
sequential_9/dense_37/MatMulMatMul)sequential_9/dropout_37/Identity:output:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/MatMul�
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp�
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/BiasAdd�
sequential_9/dense_37/ReluRelu&sequential_9/dense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/Relu�
 sequential_9/dropout_38/IdentityIdentity(sequential_9/dense_37/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_38/Identity�
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp�
sequential_9/dense_38/MatMulMatMul)sequential_9/dropout_38/Identity:output:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/MatMul�
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp�
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/BiasAdd�
sequential_9/dense_38/ReluRelu&sequential_9/dense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/Relu�
 sequential_9/dropout_39/IdentityIdentity(sequential_9/dense_38/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_39/Identity�
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_39/MatMul/ReadVariableOp�
dense_39/MatMulMatMul)sequential_9/dropout_39/Identity:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/MatMul�
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp�
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_39/Softmax�
IdentityIdentitydense_39/Softmax:softmax:0 ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp-sequential_9/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_36/Conv2D/ReadVariableOp,sequential_9/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp-sequential_9/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_37/Conv2D/ReadVariableOp,sequential_9/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp-sequential_9/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_38/Conv2D/ReadVariableOp,sequential_9/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp-sequential_9/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_39/Conv2D/ReadVariableOp,sequential_9/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_9_layer_call_fn_1729465

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
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_17264132
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
*__inference_dense_38_layer_call_fn_1729771

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
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_17267352
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
�
�
+__inference_conv2d_36_layer_call_fn_1729523

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
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_17265822
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
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1729357

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
�
i
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1726515

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
�
�
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1729514

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
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
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�>
�	
A__inference_CNN3_layer_call_and_return_conditional_losses_1727668

inputs"
sequential_9_1727601:"
sequential_9_1727603:"
sequential_9_1727605:"
sequential_9_1727607:.
sequential_9_1727609: "
sequential_9_1727611: /
sequential_9_1727613: �#
sequential_9_1727615:	�0
sequential_9_1727617:��#
sequential_9_1727619:	�0
sequential_9_1727621:��#
sequential_9_1727623:	�(
sequential_9_1727625:
�@�#
sequential_9_1727627:	�(
sequential_9_1727629:
��#
sequential_9_1727631:	�(
sequential_9_1727633:
��#
sequential_9_1727635:	�#
dense_39_1727638:	�
dense_39_1727640:
identity��2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp� dense_39/StatefulPartitionedCall�$sequential_9/StatefulPartitionedCall�
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_1727601sequential_9_1727603sequential_9_1727605sequential_9_1727607sequential_9_1727609sequential_9_1727611sequential_9_1727613sequential_9_1727615sequential_9_1727617sequential_9_1727619sequential_9_1727621sequential_9_1727623sequential_9_1727625sequential_9_1727627sequential_9_1727629sequential_9_1727631sequential_9_1727633sequential_9_1727635*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_17271682&
$sequential_9/StatefulPartitionedCall�
 dense_39/StatefulPartitionedCallStatefulPartitionedCall-sequential_9/StatefulPartitionedCall:output:0dense_39_1727638dense_39_1727640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_39_layer_call_and_return_conditional_losses_17274672"
 dense_39/StatefulPartitionedCall�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1727609*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1727625* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1727629* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1727633* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentity)dense_39/StatefulPartitionedCall:output:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp!^dense_39/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1727973
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__wrapped_model_17263472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
f
G__inference_dropout_38_layer_call_and_return_conditional_losses_1729729

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
�
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1729554

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
:����������*
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
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1729574

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
:���������		�*
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
:���������		�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������		�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
f
G__inference_dropout_36_layer_call_and_return_conditional_losses_1726937

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_37_layer_call_fn_1729675

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_17266862
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
�
F
*__inference_lambda_9_layer_call_fn_1729367

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_17270402
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
f
G__inference_dropout_36_layer_call_and_return_conditional_losses_1729600

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_38_layer_call_and_return_conditional_losses_1729717

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
�
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1726536

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
�
�
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1729534

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
:���������%%�*
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
:���������%%�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������%% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������%% 
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_38_layer_call_fn_1726509

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_17265032
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
�
�
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1726582

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
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
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_37_layer_call_fn_1726497

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_17264912
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
�
�
7__inference_batch_normalization_9_layer_call_fn_1729491

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
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_17270132
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
�
i
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1726479

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
�
e
G__inference_dropout_38_layer_call_and_return_conditional_losses_1726716

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
�
E__inference_dense_37_layer_call_and_return_conditional_losses_1729703

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
G__inference_dropout_39_layer_call_and_return_conditional_losses_1726832

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
�
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1726656

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_dense_36_layer_call_and_return_conditional_losses_1729644

inputs2
matmul_readvariableop_resource:
�@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�@�*
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
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
&__inference_CNN3_layer_call_fn_1728612

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_17276682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
E__inference_dense_38_layer_call_and_return_conditional_losses_1726735

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_9_layer_call_fn_1729321
lambda_9_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_17271682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_9_input
��
�
A__inference_CNN3_layer_call_and_return_conditional_losses_1728085

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_38_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_39_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	�H
4sequential_9_dense_36_matmul_readvariableop_resource:
�@�D
5sequential_9_dense_36_biasadd_readvariableop_resource:	�H
4sequential_9_dense_37_matmul_readvariableop_resource:
��D
5sequential_9_dense_37_biasadd_readvariableop_resource:	�H
4sequential_9_dense_38_matmul_readvariableop_resource:
��D
5sequential_9_dense_38_biasadd_readvariableop_resource:	�:
'dense_39_matmul_readvariableop_resource:	�6
(dense_39_biasadd_readvariableop_resource:
identity��2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�dense_39/BiasAdd/ReadVariableOp�dense_39/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�,sequential_9/conv2d_36/Conv2D/ReadVariableOp�-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�,sequential_9/conv2d_37/Conv2D/ReadVariableOp�-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�,sequential_9/conv2d_38/Conv2D/ReadVariableOp�-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�,sequential_9/conv2d_39/Conv2D/ReadVariableOp�,sequential_9/dense_36/BiasAdd/ReadVariableOp�+sequential_9/dense_36/MatMul/ReadVariableOp�,sequential_9/dense_37/BiasAdd/ReadVariableOp�+sequential_9/dense_37/MatMul/ReadVariableOp�,sequential_9/dense_38/BiasAdd/ReadVariableOp�+sequential_9/dense_38/MatMul/ReadVariableOp�
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack�
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1�
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2�
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice�
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp�
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3�
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOp�
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D�
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_36/BiasAdd�
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_36/Relu�
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool�
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOp�
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D�
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_37/BiasAdd�
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_37/Relu�
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool�
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOp�
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D�
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_38/BiasAdd�
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_38/Relu�
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool�
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOp�
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D�
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2 
sequential_9/conv2d_39/BiasAdd�
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
sequential_9/conv2d_39/Relu�
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool�
 sequential_9/dropout_36/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:����������2"
 sequential_9/dropout_36/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_36/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp�
sequential_9/dense_36/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/MatMul�
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp�
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/BiasAdd�
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/Relu�
 sequential_9/dropout_37/IdentityIdentity(sequential_9/dense_36/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_37/Identity�
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp�
sequential_9/dense_37/MatMulMatMul)sequential_9/dropout_37/Identity:output:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/MatMul�
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp�
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/BiasAdd�
sequential_9/dense_37/ReluRelu&sequential_9/dense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/Relu�
 sequential_9/dropout_38/IdentityIdentity(sequential_9/dense_37/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_38/Identity�
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp�
sequential_9/dense_38/MatMulMatMul)sequential_9/dropout_38/Identity:output:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/MatMul�
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp�
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/BiasAdd�
sequential_9/dense_38/ReluRelu&sequential_9/dense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/Relu�
 sequential_9/dropout_39/IdentityIdentity(sequential_9/dense_38/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_39/Identity�
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_39/MatMul/ReadVariableOp�
dense_39/MatMulMatMul)sequential_9/dropout_39/Identity:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/MatMul�
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp�
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_39/Softmax�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�

IdentityIdentitydense_39/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp-sequential_9/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_36/Conv2D/ReadVariableOp,sequential_9/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp-sequential_9/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_37/Conv2D/ReadVariableOp,sequential_9/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp-sequential_9/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_38/Conv2D/ReadVariableOp,sequential_9/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp-sequential_9/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_39/Conv2D/ReadVariableOp,sequential_9/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
.__inference_sequential_9_layer_call_fn_1729280

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_17271682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1726618

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
:����������*
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
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_37_layer_call_fn_1729543

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
:���������%%�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_17266002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������%% : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������%% 
 
_user_specified_nameinputs
��
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1728919

inputs;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_36_conv2d_readvariableop_resource: 7
)conv2d_36_biasadd_readvariableop_resource: C
(conv2d_37_conv2d_readvariableop_resource: �8
)conv2d_37_biasadd_readvariableop_resource:	�D
(conv2d_38_conv2d_readvariableop_resource:��8
)conv2d_38_biasadd_readvariableop_resource:	�D
(conv2d_39_conv2d_readvariableop_resource:��8
)conv2d_39_biasadd_readvariableop_resource:	�;
'dense_36_matmul_readvariableop_resource:
�@�7
(dense_36_biasadd_readvariableop_resource:	�;
'dense_37_matmul_readvariableop_resource:
��7
(dense_37_biasadd_readvariableop_resource:	�;
'dense_38_matmul_readvariableop_resource:
��7
(dense_38_biasadd_readvariableop_resource:	�
identity��$batch_normalization_9/AssignNewValue�&batch_normalization_9/AssignNewValue_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1� conv2d_36/BiasAdd/ReadVariableOp�conv2d_36/Conv2D/ReadVariableOp�2conv2d_36/kernel/Regularizer/Square/ReadVariableOp� conv2d_37/BiasAdd/ReadVariableOp�conv2d_37/Conv2D/ReadVariableOp� conv2d_38/BiasAdd/ReadVariableOp�conv2d_38/Conv2D/ReadVariableOp� conv2d_39/BiasAdd/ReadVariableOp�conv2d_39/Conv2D/ReadVariableOp�dense_36/BiasAdd/ReadVariableOp�dense_36/MatMul/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�dense_37/BiasAdd/ReadVariableOp�dense_37/MatMul/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�dense_38/BiasAdd/ReadVariableOp�dense_38/MatMul/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�
lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_9/strided_slice/stack�
lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_9/strided_slice/stack_1�
lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_9/strided_slice/stack_2�
lambda_9/strided_sliceStridedSliceinputs%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_9/strided_slice�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOp�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_9/FusedBatchNormV3�
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue�
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1�
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOp�
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_36/Conv2D�
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp�
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_36/Relu�
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool�
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_37/Conv2D/ReadVariableOp�
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_37/Conv2D�
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp�
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_37/Relu�
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool�
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_38/Conv2D/ReadVariableOp�
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_38/Conv2D�
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp�
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_38/Relu�
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool�
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_39/Conv2D/ReadVariableOp�
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
conv2d_39/Conv2D�
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp�
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
conv2d_39/Relu�
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPooly
dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_36/dropout/Const�
dropout_36/dropout/MulMul!max_pooling2d_39/MaxPool:output:0!dropout_36/dropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout_36/dropout/Mul�
dropout_36/dropout/ShapeShape!max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_36/dropout/Shape�
/dropout_36/dropout/random_uniform/RandomUniformRandomUniform!dropout_36/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype021
/dropout_36/dropout/random_uniform/RandomUniform�
!dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_36/dropout/GreaterEqual/y�
dropout_36/dropout/GreaterEqualGreaterEqual8dropout_36/dropout/random_uniform/RandomUniform:output:0*dropout_36/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2!
dropout_36/dropout/GreaterEqual�
dropout_36/dropout/CastCast#dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout_36/dropout/Cast�
dropout_36/dropout/Mul_1Muldropout_36/dropout/Mul:z:0dropout_36/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout_36/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_36/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2
flatten_9/Reshape�
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02 
dense_36/MatMul/ReadVariableOp�
dense_36/MatMulMatMulflatten_9/Reshape:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_36/MatMul�
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_36/BiasAdd/ReadVariableOp�
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_36/Reluy
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_37/dropout/Const�
dropout_37/dropout/MulMuldense_36/Relu:activations:0!dropout_37/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_37/dropout/Mul
dropout_37/dropout/ShapeShapedense_36/Relu:activations:0*
T0*
_output_shapes
:2
dropout_37/dropout/Shape�
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_37/dropout/random_uniform/RandomUniform�
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_37/dropout/GreaterEqual/y�
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_37/dropout/GreaterEqual�
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_37/dropout/Cast�
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_37/dropout/Mul_1�
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_37/MatMul/ReadVariableOp�
dense_37/MatMulMatMuldropout_37/dropout/Mul_1:z:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_37/MatMul�
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_37/BiasAdd/ReadVariableOp�
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_37/BiasAddt
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_37/Reluy
dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_38/dropout/Const�
dropout_38/dropout/MulMuldense_37/Relu:activations:0!dropout_38/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_38/dropout/Mul
dropout_38/dropout/ShapeShapedense_37/Relu:activations:0*
T0*
_output_shapes
:2
dropout_38/dropout/Shape�
/dropout_38/dropout/random_uniform/RandomUniformRandomUniform!dropout_38/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_38/dropout/random_uniform/RandomUniform�
!dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_38/dropout/GreaterEqual/y�
dropout_38/dropout/GreaterEqualGreaterEqual8dropout_38/dropout/random_uniform/RandomUniform:output:0*dropout_38/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_38/dropout/GreaterEqual�
dropout_38/dropout/CastCast#dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_38/dropout/Cast�
dropout_38/dropout/Mul_1Muldropout_38/dropout/Mul:z:0dropout_38/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_38/dropout/Mul_1�
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_38/MatMul/ReadVariableOp�
dense_38/MatMulMatMuldropout_38/dropout/Mul_1:z:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_38/MatMul�
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_38/BiasAdd/ReadVariableOp�
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_38/BiasAddt
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_38/Reluy
dropout_39/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_39/dropout/Const�
dropout_39/dropout/MulMuldense_38/Relu:activations:0!dropout_39/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_39/dropout/Mul
dropout_39/dropout/ShapeShapedense_38/Relu:activations:0*
T0*
_output_shapes
:2
dropout_39/dropout/Shape�
/dropout_39/dropout/random_uniform/RandomUniformRandomUniform!dropout_39/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_39/dropout/random_uniform/RandomUniform�
!dropout_39/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_39/dropout/GreaterEqual/y�
dropout_39/dropout/GreaterEqualGreaterEqual8dropout_39/dropout/random_uniform/RandomUniform:output:0*dropout_39/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_39/dropout/GreaterEqual�
dropout_39/dropout/CastCast#dropout_39/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_39/dropout/Cast�
dropout_39/dropout/Mul_1Muldropout_39/dropout/Mul:z:0dropout_39/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_39/dropout/Mul_1�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentitydropout_39/dropout/Mul_1:z:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
e
,__inference_dropout_36_layer_call_fn_1729610

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_17269372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_dense_37_layer_call_and_return_conditional_losses_1726705

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�}
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1727168

inputs+
batch_normalization_9_1727090:+
batch_normalization_9_1727092:+
batch_normalization_9_1727094:+
batch_normalization_9_1727096:+
conv2d_36_1727099: 
conv2d_36_1727101: ,
conv2d_37_1727105: � 
conv2d_37_1727107:	�-
conv2d_38_1727111:�� 
conv2d_38_1727113:	�-
conv2d_39_1727117:�� 
conv2d_39_1727119:	�$
dense_36_1727125:
�@�
dense_36_1727127:	�$
dense_37_1727131:
��
dense_37_1727133:	�$
dense_38_1727137:
��
dense_38_1727139:	�
identity��-batch_normalization_9/StatefulPartitionedCall�!conv2d_36/StatefulPartitionedCall�2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�!conv2d_37/StatefulPartitionedCall�!conv2d_38/StatefulPartitionedCall�!conv2d_39/StatefulPartitionedCall� dense_36/StatefulPartitionedCall�1dense_36/kernel/Regularizer/Square/ReadVariableOp� dense_37/StatefulPartitionedCall�1dense_37/kernel/Regularizer/Square/ReadVariableOp� dense_38/StatefulPartitionedCall�1dense_38/kernel/Regularizer/Square/ReadVariableOp�"dropout_36/StatefulPartitionedCall�"dropout_37/StatefulPartitionedCall�"dropout_38/StatefulPartitionedCall�"dropout_39/StatefulPartitionedCall�
lambda_9/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_17270402
lambda_9/PartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall!lambda_9/PartitionedCall:output:0batch_normalization_9_1727090batch_normalization_9_1727092batch_normalization_9_1727094batch_normalization_9_1727096*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_17270132/
-batch_normalization_9/StatefulPartitionedCall�
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_36_1727099conv2d_36_1727101*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_17265822#
!conv2d_36/StatefulPartitionedCall�
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������%% * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_17264792"
 max_pooling2d_36/PartitionedCall�
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_37_1727105conv2d_37_1727107*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_17266002#
!conv2d_37/StatefulPartitionedCall�
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_17264912"
 max_pooling2d_37/PartitionedCall�
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_38_1727111conv2d_38_1727113*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_17266182#
!conv2d_38/StatefulPartitionedCall�
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_17265032"
 max_pooling2d_38/PartitionedCall�
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_1727117conv2d_39_1727119*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_17266362#
!conv2d_39/StatefulPartitionedCall�
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_17265152"
 max_pooling2d_39/PartitionedCall�
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_17269372$
"dropout_36/StatefulPartitionedCall�
flatten_9/PartitionedCallPartitionedCall+dropout_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_17266562
flatten_9/PartitionedCall�
 dense_36/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_36_1727125dense_36_1727127*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_36_layer_call_and_return_conditional_losses_17266752"
 dense_36/StatefulPartitionedCall�
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_17268982$
"dropout_37/StatefulPartitionedCall�
 dense_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_37_1727131dense_37_1727133*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_37_layer_call_and_return_conditional_losses_17267052"
 dense_37/StatefulPartitionedCall�
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_17268652$
"dropout_38/StatefulPartitionedCall�
 dense_38/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_38_1727137dense_38_1727139*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_17267352"
 dense_38/StatefulPartitionedCall�
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0#^dropout_38/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_17268322$
"dropout_39/StatefulPartitionedCall�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_36_1727099*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_1727125* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_1727131* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_1727137* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentity+dropout_39/StatefulPartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall2^dense_36/kernel/Regularizer/Square/ReadVariableOp!^dense_37/StatefulPartitionedCall2^dense_37/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall2^dense_38/kernel/Regularizer/Square/ReadVariableOp#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
E__inference_dense_36_layer_call_and_return_conditional_losses_1726675

inputs2
matmul_readvariableop_resource:
�@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�@�*
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
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_9_layer_call_fn_1729452

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
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_17263692
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
�
i
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1726503

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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729403

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
�
&__inference_CNN3_layer_call_fn_1728567

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_17274982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1727013

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
�
F
*__inference_lambda_9_layer_call_fn_1729362

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_17265362
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
�
e
G__inference_dropout_37_layer_call_and_return_conditional_losses_1726686

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
G__inference_dropout_38_layer_call_and_return_conditional_losses_1726865

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
�
�
*__inference_dense_39_layer_call_fn_1729341

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
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_39_layer_call_and_return_conditional_losses_17274672
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
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729421

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
�
�
*__inference_dense_36_layer_call_fn_1729653

inputs
unknown:
�@�
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
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_36_layer_call_and_return_conditional_losses_17266752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������@
 
_user_specified_nameinputs
�

�
E__inference_dense_39_layer_call_and_return_conditional_losses_1729332

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
�
�
__inference_loss_fn_0_1729809U
;conv2d_36_kernel_regularizer_square_readvariableop_resource: 
identity��2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_36_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
IdentityIdentity$conv2d_36/kernel/Regularizer/mul:z:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_1_1729820N
:dense_36_kernel_regularizer_square_readvariableop_resource:
�@�
identity��1dense_36/kernel/Regularizer/Square/ReadVariableOp�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_36_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
IdentityIdentity#dense_36/kernel/Regularizer/mul:z:02^dense_36/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1726369

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
�
�
.__inference_sequential_9_layer_call_fn_1729198
lambda_9_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_17267732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_9_input
�
�
+__inference_conv2d_39_layer_call_fn_1729583

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
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_17266362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������		�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
��
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1729024
lambda_9_input;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_36_conv2d_readvariableop_resource: 7
)conv2d_36_biasadd_readvariableop_resource: C
(conv2d_37_conv2d_readvariableop_resource: �8
)conv2d_37_biasadd_readvariableop_resource:	�D
(conv2d_38_conv2d_readvariableop_resource:��8
)conv2d_38_biasadd_readvariableop_resource:	�D
(conv2d_39_conv2d_readvariableop_resource:��8
)conv2d_39_biasadd_readvariableop_resource:	�;
'dense_36_matmul_readvariableop_resource:
�@�7
(dense_36_biasadd_readvariableop_resource:	�;
'dense_37_matmul_readvariableop_resource:
��7
(dense_37_biasadd_readvariableop_resource:	�;
'dense_38_matmul_readvariableop_resource:
��7
(dense_38_biasadd_readvariableop_resource:	�
identity��5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1� conv2d_36/BiasAdd/ReadVariableOp�conv2d_36/Conv2D/ReadVariableOp�2conv2d_36/kernel/Regularizer/Square/ReadVariableOp� conv2d_37/BiasAdd/ReadVariableOp�conv2d_37/Conv2D/ReadVariableOp� conv2d_38/BiasAdd/ReadVariableOp�conv2d_38/Conv2D/ReadVariableOp� conv2d_39/BiasAdd/ReadVariableOp�conv2d_39/Conv2D/ReadVariableOp�dense_36/BiasAdd/ReadVariableOp�dense_36/MatMul/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�dense_37/BiasAdd/ReadVariableOp�dense_37/MatMul/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�dense_38/BiasAdd/ReadVariableOp�dense_38/MatMul/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�
lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_9/strided_slice/stack�
lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_9/strided_slice/stack_1�
lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_9/strided_slice/stack_2�
lambda_9/strided_sliceStridedSlicelambda_9_input%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_9/strided_slice�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOp�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3�
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOp�
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_36/Conv2D�
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp�
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_36/Relu�
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool�
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_37/Conv2D/ReadVariableOp�
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_37/Conv2D�
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp�
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_37/Relu�
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool�
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_38/Conv2D/ReadVariableOp�
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_38/Conv2D�
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp�
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_38/Relu�
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool�
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_39/Conv2D/ReadVariableOp�
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
conv2d_39/Conv2D�
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp�
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
conv2d_39/Relu�
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPool�
dropout_36/IdentityIdentity!max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:����������2
dropout_36/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_36/Identity:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2
flatten_9/Reshape�
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02 
dense_36/MatMul/ReadVariableOp�
dense_36/MatMulMatMulflatten_9/Reshape:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_36/MatMul�
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_36/BiasAdd/ReadVariableOp�
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_36/Relu�
dropout_37/IdentityIdentitydense_36/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_37/Identity�
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_37/MatMul/ReadVariableOp�
dense_37/MatMulMatMuldropout_37/Identity:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_37/MatMul�
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_37/BiasAdd/ReadVariableOp�
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_37/BiasAddt
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_37/Relu�
dropout_38/IdentityIdentitydense_37/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_38/Identity�
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_38/MatMul/ReadVariableOp�
dense_38/MatMulMatMuldropout_38/Identity:output:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_38/MatMul�
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_38/BiasAdd/ReadVariableOp�
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_38/BiasAddt
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_38/Relu�
dropout_39/IdentityIdentitydense_38/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_39/Identity�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentitydropout_39/Identity:output:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_9_input
�
�
7__inference_batch_normalization_9_layer_call_fn_1729478

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
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_17265552
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
�>
�	
A__inference_CNN3_layer_call_and_return_conditional_losses_1727498

inputs"
sequential_9_1727419:"
sequential_9_1727421:"
sequential_9_1727423:"
sequential_9_1727425:.
sequential_9_1727427: "
sequential_9_1727429: /
sequential_9_1727431: �#
sequential_9_1727433:	�0
sequential_9_1727435:��#
sequential_9_1727437:	�0
sequential_9_1727439:��#
sequential_9_1727441:	�(
sequential_9_1727443:
�@�#
sequential_9_1727445:	�(
sequential_9_1727447:
��#
sequential_9_1727449:	�(
sequential_9_1727451:
��#
sequential_9_1727453:	�#
dense_39_1727468:	�
dense_39_1727470:
identity��2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp� dense_39/StatefulPartitionedCall�$sequential_9/StatefulPartitionedCall�
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_1727419sequential_9_1727421sequential_9_1727423sequential_9_1727425sequential_9_1727427sequential_9_1727429sequential_9_1727431sequential_9_1727433sequential_9_1727435sequential_9_1727437sequential_9_1727439sequential_9_1727441sequential_9_1727443sequential_9_1727445sequential_9_1727447sequential_9_1727449sequential_9_1727451sequential_9_1727453*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_17267732&
$sequential_9/StatefulPartitionedCall�
 dense_39/StatefulPartitionedCallStatefulPartitionedCall-sequential_9/StatefulPartitionedCall:output:0dense_39_1727468dense_39_1727470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_39_layer_call_and_return_conditional_losses_17274672"
 dense_39/StatefulPartitionedCall�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1727427*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1727443* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1727447* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1727451* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentity)dense_39/StatefulPartitionedCall:output:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp!^dense_39/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
��
�
A__inference_CNN3_layer_call_and_return_conditional_losses_1728477
input_1H
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_38_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_39_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	�H
4sequential_9_dense_36_matmul_readvariableop_resource:
�@�D
5sequential_9_dense_36_biasadd_readvariableop_resource:	�H
4sequential_9_dense_37_matmul_readvariableop_resource:
��D
5sequential_9_dense_37_biasadd_readvariableop_resource:	�H
4sequential_9_dense_38_matmul_readvariableop_resource:
��D
5sequential_9_dense_38_biasadd_readvariableop_resource:	�:
'dense_39_matmul_readvariableop_resource:	�6
(dense_39_biasadd_readvariableop_resource:
identity��2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�dense_39/BiasAdd/ReadVariableOp�dense_39/MatMul/ReadVariableOp�1sequential_9/batch_normalization_9/AssignNewValue�3sequential_9/batch_normalization_9/AssignNewValue_1�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�,sequential_9/conv2d_36/Conv2D/ReadVariableOp�-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�,sequential_9/conv2d_37/Conv2D/ReadVariableOp�-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�,sequential_9/conv2d_38/Conv2D/ReadVariableOp�-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�,sequential_9/conv2d_39/Conv2D/ReadVariableOp�,sequential_9/dense_36/BiasAdd/ReadVariableOp�+sequential_9/dense_36/MatMul/ReadVariableOp�,sequential_9/dense_37/BiasAdd/ReadVariableOp�+sequential_9/dense_37/MatMul/ReadVariableOp�,sequential_9/dense_38/BiasAdd/ReadVariableOp�+sequential_9/dense_38/MatMul/ReadVariableOp�
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack�
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1�
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2�
#sequential_9/lambda_9/strided_sliceStridedSliceinput_12sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice�
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp�
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<25
3sequential_9/batch_normalization_9/FusedBatchNormV3�
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_9/batch_normalization_9/AssignNewValue�
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_9/batch_normalization_9/AssignNewValue_1�
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOp�
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D�
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_36/BiasAdd�
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_36/Relu�
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool�
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOp�
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D�
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_37/BiasAdd�
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_37/Relu�
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool�
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOp�
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D�
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_38/BiasAdd�
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_38/Relu�
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool�
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOp�
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D�
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2 
sequential_9/conv2d_39/BiasAdd�
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
sequential_9/conv2d_39/Relu�
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool�
%sequential_9/dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2'
%sequential_9/dropout_36/dropout/Const�
#sequential_9/dropout_36/dropout/MulMul.sequential_9/max_pooling2d_39/MaxPool:output:0.sequential_9/dropout_36/dropout/Const:output:0*
T0*0
_output_shapes
:����������2%
#sequential_9/dropout_36/dropout/Mul�
%sequential_9/dropout_36/dropout/ShapeShape.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_36/dropout/Shape�
<sequential_9/dropout_36/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_36/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_36/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=20
.sequential_9/dropout_36/dropout/GreaterEqual/y�
,sequential_9/dropout_36/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_36/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_36/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2.
,sequential_9/dropout_36/dropout/GreaterEqual�
$sequential_9/dropout_36/dropout/CastCast0sequential_9/dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2&
$sequential_9/dropout_36/dropout/Cast�
%sequential_9/dropout_36/dropout/Mul_1Mul'sequential_9/dropout_36/dropout/Mul:z:0(sequential_9/dropout_36/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2'
%sequential_9/dropout_36/dropout/Mul_1�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_36/dropout/Mul_1:z:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp�
sequential_9/dense_36/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/MatMul�
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp�
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/BiasAdd�
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/Relu�
%sequential_9/dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_37/dropout/Const�
#sequential_9/dropout_37/dropout/MulMul(sequential_9/dense_36/Relu:activations:0.sequential_9/dropout_37/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_37/dropout/Mul�
%sequential_9/dropout_37/dropout/ShapeShape(sequential_9/dense_36/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_37/dropout/Shape�
<sequential_9/dropout_37/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_37/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_37/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_37/dropout/GreaterEqual/y�
,sequential_9/dropout_37/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_37/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_37/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_37/dropout/GreaterEqual�
$sequential_9/dropout_37/dropout/CastCast0sequential_9/dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_37/dropout/Cast�
%sequential_9/dropout_37/dropout/Mul_1Mul'sequential_9/dropout_37/dropout/Mul:z:0(sequential_9/dropout_37/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_37/dropout/Mul_1�
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp�
sequential_9/dense_37/MatMulMatMul)sequential_9/dropout_37/dropout/Mul_1:z:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/MatMul�
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp�
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/BiasAdd�
sequential_9/dense_37/ReluRelu&sequential_9/dense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/Relu�
%sequential_9/dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_38/dropout/Const�
#sequential_9/dropout_38/dropout/MulMul(sequential_9/dense_37/Relu:activations:0.sequential_9/dropout_38/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_38/dropout/Mul�
%sequential_9/dropout_38/dropout/ShapeShape(sequential_9/dense_37/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_38/dropout/Shape�
<sequential_9/dropout_38/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_38/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_38/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_38/dropout/GreaterEqual/y�
,sequential_9/dropout_38/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_38/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_38/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_38/dropout/GreaterEqual�
$sequential_9/dropout_38/dropout/CastCast0sequential_9/dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_38/dropout/Cast�
%sequential_9/dropout_38/dropout/Mul_1Mul'sequential_9/dropout_38/dropout/Mul:z:0(sequential_9/dropout_38/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_38/dropout/Mul_1�
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp�
sequential_9/dense_38/MatMulMatMul)sequential_9/dropout_38/dropout/Mul_1:z:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/MatMul�
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp�
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/BiasAdd�
sequential_9/dense_38/ReluRelu&sequential_9/dense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/Relu�
%sequential_9/dropout_39/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_39/dropout/Const�
#sequential_9/dropout_39/dropout/MulMul(sequential_9/dense_38/Relu:activations:0.sequential_9/dropout_39/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_39/dropout/Mul�
%sequential_9/dropout_39/dropout/ShapeShape(sequential_9/dense_38/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_39/dropout/Shape�
<sequential_9/dropout_39/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_39/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_39/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_39/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_39/dropout/GreaterEqual/y�
,sequential_9/dropout_39/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_39/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_39/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_39/dropout/GreaterEqual�
$sequential_9/dropout_39/dropout/CastCast0sequential_9/dropout_39/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_39/dropout/Cast�
%sequential_9/dropout_39/dropout/Mul_1Mul'sequential_9/dropout_39/dropout/Mul:z:0(sequential_9/dropout_39/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_39/dropout/Mul_1�
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_39/MatMul/ReadVariableOp�
dense_39/MatMulMatMul)sequential_9/dropout_39/dropout/Mul_1:z:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/MatMul�
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp�
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_39/Softmax�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�

IdentityIdentitydense_39/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_12�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp-sequential_9/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_36/Conv2D/ReadVariableOp,sequential_9/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp-sequential_9/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_37/Conv2D/ReadVariableOp,sequential_9/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp-sequential_9/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_38/Conv2D/ReadVariableOp,sequential_9/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp-sequential_9/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_39/Conv2D/ReadVariableOp,sequential_9/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�|
�
 __inference__traced_save_1730060
file_prefix.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop/
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop/
+savev2_conv2d_39_kernel_read_readvariableop-
)savev2_conv2d_39_bias_read_readvariableop.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_39_kernel_m_read_readvariableop3
/savev2_adam_dense_39_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_m_read_readvariableop6
2savev2_adam_conv2d_36_kernel_m_read_readvariableop4
0savev2_adam_conv2d_36_bias_m_read_readvariableop6
2savev2_adam_conv2d_37_kernel_m_read_readvariableop4
0savev2_adam_conv2d_37_bias_m_read_readvariableop6
2savev2_adam_conv2d_38_kernel_m_read_readvariableop4
0savev2_adam_conv2d_38_bias_m_read_readvariableop6
2savev2_adam_conv2d_39_kernel_m_read_readvariableop4
0savev2_adam_conv2d_39_bias_m_read_readvariableop5
1savev2_adam_dense_36_kernel_m_read_readvariableop3
/savev2_adam_dense_36_bias_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop5
1savev2_adam_dense_38_kernel_m_read_readvariableop3
/savev2_adam_dense_38_bias_m_read_readvariableop5
1savev2_adam_dense_39_kernel_v_read_readvariableop3
/savev2_adam_dense_39_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_v_read_readvariableop6
2savev2_adam_conv2d_36_kernel_v_read_readvariableop4
0savev2_adam_conv2d_36_bias_v_read_readvariableop6
2savev2_adam_conv2d_37_kernel_v_read_readvariableop4
0savev2_adam_conv2d_37_bias_v_read_readvariableop6
2savev2_adam_conv2d_38_kernel_v_read_readvariableop4
0savev2_adam_conv2d_38_bias_v_read_readvariableop6
2savev2_adam_conv2d_39_kernel_v_read_readvariableop4
0savev2_adam_conv2d_39_bias_v_read_readvariableop5
1savev2_adam_dense_36_kernel_v_read_readvariableop3
/savev2_adam_dense_36_bias_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop5
1savev2_adam_dense_38_kernel_v_read_readvariableop3
/savev2_adam_dense_38_bias_v_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*�
value�B�BB)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*�
value�B�BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableop*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_39_kernel_m_read_readvariableop/savev2_adam_dense_39_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop2savev2_adam_conv2d_36_kernel_m_read_readvariableop0savev2_adam_conv2d_36_bias_m_read_readvariableop2savev2_adam_conv2d_37_kernel_m_read_readvariableop0savev2_adam_conv2d_37_bias_m_read_readvariableop2savev2_adam_conv2d_38_kernel_m_read_readvariableop0savev2_adam_conv2d_38_bias_m_read_readvariableop2savev2_adam_conv2d_39_kernel_m_read_readvariableop0savev2_adam_conv2d_39_bias_m_read_readvariableop1savev2_adam_dense_36_kernel_m_read_readvariableop/savev2_adam_dense_36_bias_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop1savev2_adam_dense_38_kernel_m_read_readvariableop/savev2_adam_dense_38_bias_m_read_readvariableop1savev2_adam_dense_39_kernel_v_read_readvariableop/savev2_adam_dense_39_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop2savev2_adam_conv2d_36_kernel_v_read_readvariableop0savev2_adam_conv2d_36_bias_v_read_readvariableop2savev2_adam_conv2d_37_kernel_v_read_readvariableop0savev2_adam_conv2d_37_bias_v_read_readvariableop2savev2_adam_conv2d_38_kernel_v_read_readvariableop0savev2_adam_conv2d_38_bias_v_read_readvariableop2savev2_adam_conv2d_39_kernel_v_read_readvariableop0savev2_adam_conv2d_39_bias_v_read_readvariableop1savev2_adam_dense_36_kernel_v_read_readvariableop/savev2_adam_dense_36_bias_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableop1savev2_adam_dense_38_kernel_v_read_readvariableop/savev2_adam_dense_38_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *P
dtypesF
D2B	2
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:: : : : : ::::: : : �:�:��:�:��:�:
�@�:�:
��:�:
��:�: : : : :	�:::: : : �:�:��:�:��:�:
�@�:�:
��:�:
��:�:	�:::: : : �:�:��:�:��:�:
�@�:�:
��:�:
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
: :-)
'
_output_shapes
: �:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:&"
 
_output_shapes
:
�@�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
: : #

_output_shapes
: :-$)
'
_output_shapes
: �:!%

_output_shapes	
:�:.&*
(
_output_shapes
:��:!'

_output_shapes	
:�:.(*
(
_output_shapes
:��:!)

_output_shapes	
:�:&*"
 
_output_shapes
:
�@�:!+

_output_shapes	
:�:&,"
 
_output_shapes
:
��:!-

_output_shapes	
:�:&."
 
_output_shapes
:
��:!/

_output_shapes	
:�:%0!

_output_shapes
:	�: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
: : 5

_output_shapes
: :-6)
'
_output_shapes
: �:!7

_output_shapes	
:�:.8*
(
_output_shapes
:��:!9

_output_shapes	
:�:.:*
(
_output_shapes
:��:!;

_output_shapes	
:�:&<"
 
_output_shapes
:
�@�:!=

_output_shapes	
:�:&>"
 
_output_shapes
:
��:!?

_output_shapes	
:�:&@"
 
_output_shapes
:
��:!A

_output_shapes	
:�:B

_output_shapes
: 
�
�
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1726600

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
:���������%%�*
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
:���������%%�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������%%�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������%% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������%% 
 
_user_specified_nameinputs
�

�
E__inference_dense_39_layer_call_and_return_conditional_losses_1727467

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
�
f
G__inference_dropout_37_layer_call_and_return_conditional_losses_1726898

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
�
H
,__inference_dropout_36_layer_call_fn_1729605

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_17266482
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
G__inference_dropout_37_layer_call_and_return_conditional_losses_1729670

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
�
e
,__inference_dropout_38_layer_call_fn_1729739

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_17268652
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
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1727040

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
�
e
G__inference_dropout_39_layer_call_and_return_conditional_losses_1726746

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
�
E__inference_dense_38_layer_call_and_return_conditional_losses_1729762

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1729157
lambda_9_input;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_36_conv2d_readvariableop_resource: 7
)conv2d_36_biasadd_readvariableop_resource: C
(conv2d_37_conv2d_readvariableop_resource: �8
)conv2d_37_biasadd_readvariableop_resource:	�D
(conv2d_38_conv2d_readvariableop_resource:��8
)conv2d_38_biasadd_readvariableop_resource:	�D
(conv2d_39_conv2d_readvariableop_resource:��8
)conv2d_39_biasadd_readvariableop_resource:	�;
'dense_36_matmul_readvariableop_resource:
�@�7
(dense_36_biasadd_readvariableop_resource:	�;
'dense_37_matmul_readvariableop_resource:
��7
(dense_37_biasadd_readvariableop_resource:	�;
'dense_38_matmul_readvariableop_resource:
��7
(dense_38_biasadd_readvariableop_resource:	�
identity��$batch_normalization_9/AssignNewValue�&batch_normalization_9/AssignNewValue_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1� conv2d_36/BiasAdd/ReadVariableOp�conv2d_36/Conv2D/ReadVariableOp�2conv2d_36/kernel/Regularizer/Square/ReadVariableOp� conv2d_37/BiasAdd/ReadVariableOp�conv2d_37/Conv2D/ReadVariableOp� conv2d_38/BiasAdd/ReadVariableOp�conv2d_38/Conv2D/ReadVariableOp� conv2d_39/BiasAdd/ReadVariableOp�conv2d_39/Conv2D/ReadVariableOp�dense_36/BiasAdd/ReadVariableOp�dense_36/MatMul/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�dense_37/BiasAdd/ReadVariableOp�dense_37/MatMul/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�dense_38/BiasAdd/ReadVariableOp�dense_38/MatMul/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�
lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_9/strided_slice/stack�
lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_9/strided_slice/stack_1�
lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_9/strided_slice/stack_2�
lambda_9/strided_sliceStridedSlicelambda_9_input%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_9/strided_slice�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOp�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_9/FusedBatchNormV3�
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue�
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1�
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOp�
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_36/Conv2D�
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp�
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_36/Relu�
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool�
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_37/Conv2D/ReadVariableOp�
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_37/Conv2D�
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp�
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_37/Relu�
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool�
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_38/Conv2D/ReadVariableOp�
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_38/Conv2D�
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp�
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_38/Relu�
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool�
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_39/Conv2D/ReadVariableOp�
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
conv2d_39/Conv2D�
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp�
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
conv2d_39/Relu�
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPooly
dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_36/dropout/Const�
dropout_36/dropout/MulMul!max_pooling2d_39/MaxPool:output:0!dropout_36/dropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout_36/dropout/Mul�
dropout_36/dropout/ShapeShape!max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_36/dropout/Shape�
/dropout_36/dropout/random_uniform/RandomUniformRandomUniform!dropout_36/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype021
/dropout_36/dropout/random_uniform/RandomUniform�
!dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_36/dropout/GreaterEqual/y�
dropout_36/dropout/GreaterEqualGreaterEqual8dropout_36/dropout/random_uniform/RandomUniform:output:0*dropout_36/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2!
dropout_36/dropout/GreaterEqual�
dropout_36/dropout/CastCast#dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout_36/dropout/Cast�
dropout_36/dropout/Mul_1Muldropout_36/dropout/Mul:z:0dropout_36/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout_36/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_36/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2
flatten_9/Reshape�
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02 
dense_36/MatMul/ReadVariableOp�
dense_36/MatMulMatMulflatten_9/Reshape:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_36/MatMul�
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_36/BiasAdd/ReadVariableOp�
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_36/Reluy
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_37/dropout/Const�
dropout_37/dropout/MulMuldense_36/Relu:activations:0!dropout_37/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_37/dropout/Mul
dropout_37/dropout/ShapeShapedense_36/Relu:activations:0*
T0*
_output_shapes
:2
dropout_37/dropout/Shape�
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_37/dropout/random_uniform/RandomUniform�
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_37/dropout/GreaterEqual/y�
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_37/dropout/GreaterEqual�
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_37/dropout/Cast�
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_37/dropout/Mul_1�
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_37/MatMul/ReadVariableOp�
dense_37/MatMulMatMuldropout_37/dropout/Mul_1:z:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_37/MatMul�
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_37/BiasAdd/ReadVariableOp�
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_37/BiasAddt
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_37/Reluy
dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_38/dropout/Const�
dropout_38/dropout/MulMuldense_37/Relu:activations:0!dropout_38/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_38/dropout/Mul
dropout_38/dropout/ShapeShapedense_37/Relu:activations:0*
T0*
_output_shapes
:2
dropout_38/dropout/Shape�
/dropout_38/dropout/random_uniform/RandomUniformRandomUniform!dropout_38/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_38/dropout/random_uniform/RandomUniform�
!dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_38/dropout/GreaterEqual/y�
dropout_38/dropout/GreaterEqualGreaterEqual8dropout_38/dropout/random_uniform/RandomUniform:output:0*dropout_38/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_38/dropout/GreaterEqual�
dropout_38/dropout/CastCast#dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_38/dropout/Cast�
dropout_38/dropout/Mul_1Muldropout_38/dropout/Mul:z:0dropout_38/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_38/dropout/Mul_1�
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_38/MatMul/ReadVariableOp�
dense_38/MatMulMatMuldropout_38/dropout/Mul_1:z:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_38/MatMul�
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_38/BiasAdd/ReadVariableOp�
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_38/BiasAddt
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_38/Reluy
dropout_39/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_39/dropout/Const�
dropout_39/dropout/MulMuldense_38/Relu:activations:0!dropout_39/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_39/dropout/Mul
dropout_39/dropout/ShapeShapedense_38/Relu:activations:0*
T0*
_output_shapes
:2
dropout_39/dropout/Shape�
/dropout_39/dropout/random_uniform/RandomUniformRandomUniform!dropout_39/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_39/dropout/random_uniform/RandomUniform�
!dropout_39/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_39/dropout/GreaterEqual/y�
dropout_39/dropout/GreaterEqualGreaterEqual8dropout_39/dropout/random_uniform/RandomUniform:output:0*dropout_39/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_39/dropout/GreaterEqual�
dropout_39/dropout/CastCast#dropout_39/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_39/dropout/Cast�
dropout_39/dropout/Mul_1Muldropout_39/dropout/Mul:z:0dropout_39/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_39/dropout/Mul_1�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentitydropout_39/dropout/Mul_1:z:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_9_input
��
�
A__inference_CNN3_layer_call_and_return_conditional_losses_1728337
input_1H
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_38_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_39_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	�H
4sequential_9_dense_36_matmul_readvariableop_resource:
�@�D
5sequential_9_dense_36_biasadd_readvariableop_resource:	�H
4sequential_9_dense_37_matmul_readvariableop_resource:
��D
5sequential_9_dense_37_biasadd_readvariableop_resource:	�H
4sequential_9_dense_38_matmul_readvariableop_resource:
��D
5sequential_9_dense_38_biasadd_readvariableop_resource:	�:
'dense_39_matmul_readvariableop_resource:	�6
(dense_39_biasadd_readvariableop_resource:
identity��2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�1dense_36/kernel/Regularizer/Square/ReadVariableOp�1dense_37/kernel/Regularizer/Square/ReadVariableOp�1dense_38/kernel/Regularizer/Square/ReadVariableOp�dense_39/BiasAdd/ReadVariableOp�dense_39/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�,sequential_9/conv2d_36/Conv2D/ReadVariableOp�-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�,sequential_9/conv2d_37/Conv2D/ReadVariableOp�-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�,sequential_9/conv2d_38/Conv2D/ReadVariableOp�-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�,sequential_9/conv2d_39/Conv2D/ReadVariableOp�,sequential_9/dense_36/BiasAdd/ReadVariableOp�+sequential_9/dense_36/MatMul/ReadVariableOp�,sequential_9/dense_37/BiasAdd/ReadVariableOp�+sequential_9/dense_37/MatMul/ReadVariableOp�,sequential_9/dense_38/BiasAdd/ReadVariableOp�+sequential_9/dense_38/MatMul/ReadVariableOp�
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack�
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1�
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2�
#sequential_9/lambda_9/strided_sliceStridedSliceinput_12sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice�
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp�
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3�
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOp�
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D�
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_36/BiasAdd�
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_36/Relu�
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool�
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOp�
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D�
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_37/BiasAdd�
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_37/Relu�
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool�
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOp�
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D�
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_38/BiasAdd�
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_38/Relu�
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool�
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOp�
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D�
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2 
sequential_9/conv2d_39/BiasAdd�
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
sequential_9/conv2d_39/Relu�
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool�
 sequential_9/dropout_36/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:����������2"
 sequential_9/dropout_36/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_36/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp�
sequential_9/dense_36/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/MatMul�
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp�
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/BiasAdd�
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/Relu�
 sequential_9/dropout_37/IdentityIdentity(sequential_9/dense_36/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_37/Identity�
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp�
sequential_9/dense_37/MatMulMatMul)sequential_9/dropout_37/Identity:output:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/MatMul�
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp�
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/BiasAdd�
sequential_9/dense_37/ReluRelu&sequential_9/dense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/Relu�
 sequential_9/dropout_38/IdentityIdentity(sequential_9/dense_37/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_38/Identity�
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp�
sequential_9/dense_38/MatMulMatMul)sequential_9/dropout_38/Identity:output:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/MatMul�
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp�
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/BiasAdd�
sequential_9/dense_38/ReluRelu&sequential_9/dense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/Relu�
 sequential_9/dropout_39/IdentityIdentity(sequential_9/dense_38/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_39/Identity�
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_39/MatMul/ReadVariableOp�
dense_39/MatMulMatMul)sequential_9/dropout_39/Identity:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/MatMul�
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp�
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_39/Softmax�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�

IdentityIdentitydense_39/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp2^dense_38/kernel/Regularizer/Square/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp-sequential_9/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_36/Conv2D/ReadVariableOp,sequential_9/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp-sequential_9/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_37/Conv2D/ReadVariableOp,sequential_9/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp-sequential_9/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_38/Conv2D/ReadVariableOp,sequential_9/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp-sequential_9/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_39/Conv2D/ReadVariableOp,sequential_9/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
e
G__inference_dropout_36_layer_call_and_return_conditional_losses_1726648

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_1729842N
:dense_38_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_38/kernel/Regularizer/Square/ReadVariableOp�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_38_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentity#dense_38/kernel/Regularizer/mul:z:02^dense_38/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729439

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
�
e
G__inference_dropout_37_layer_call_and_return_conditional_losses_1729658

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
__inference_loss_fn_2_1729831N
:dense_37_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_37/kernel/Regularizer/Square/ReadVariableOp�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_37_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
IdentityIdentity#dense_37/kernel/Regularizer/mul:z:02^dense_37/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp
�
i
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1726491

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
�
N
2__inference_max_pooling2d_36_layer_call_fn_1726485

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_17264792
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
�
H
,__inference_dropout_38_layer_call_fn_1729734

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_17267162
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
�
e
G__inference_dropout_39_layer_call_and_return_conditional_losses_1729776

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
�w
�

I__inference_sequential_9_layer_call_and_return_conditional_losses_1726773

inputs+
batch_normalization_9_1726556:+
batch_normalization_9_1726558:+
batch_normalization_9_1726560:+
batch_normalization_9_1726562:+
conv2d_36_1726583: 
conv2d_36_1726585: ,
conv2d_37_1726601: � 
conv2d_37_1726603:	�-
conv2d_38_1726619:�� 
conv2d_38_1726621:	�-
conv2d_39_1726637:�� 
conv2d_39_1726639:	�$
dense_36_1726676:
�@�
dense_36_1726678:	�$
dense_37_1726706:
��
dense_37_1726708:	�$
dense_38_1726736:
��
dense_38_1726738:	�
identity��-batch_normalization_9/StatefulPartitionedCall�!conv2d_36/StatefulPartitionedCall�2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�!conv2d_37/StatefulPartitionedCall�!conv2d_38/StatefulPartitionedCall�!conv2d_39/StatefulPartitionedCall� dense_36/StatefulPartitionedCall�1dense_36/kernel/Regularizer/Square/ReadVariableOp� dense_37/StatefulPartitionedCall�1dense_37/kernel/Regularizer/Square/ReadVariableOp� dense_38/StatefulPartitionedCall�1dense_38/kernel/Regularizer/Square/ReadVariableOp�
lambda_9/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_17265362
lambda_9/PartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall!lambda_9/PartitionedCall:output:0batch_normalization_9_1726556batch_normalization_9_1726558batch_normalization_9_1726560batch_normalization_9_1726562*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_17265552/
-batch_normalization_9/StatefulPartitionedCall�
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_36_1726583conv2d_36_1726585*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_17265822#
!conv2d_36/StatefulPartitionedCall�
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������%% * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_17264792"
 max_pooling2d_36/PartitionedCall�
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_37_1726601conv2d_37_1726603*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_17266002#
!conv2d_37/StatefulPartitionedCall�
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_17264912"
 max_pooling2d_37/PartitionedCall�
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_38_1726619conv2d_38_1726621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_17266182#
!conv2d_38/StatefulPartitionedCall�
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_17265032"
 max_pooling2d_38/PartitionedCall�
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_1726637conv2d_39_1726639*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_17266362#
!conv2d_39/StatefulPartitionedCall�
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_17265152"
 max_pooling2d_39/PartitionedCall�
dropout_36/PartitionedCallPartitionedCall)max_pooling2d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_17266482
dropout_36/PartitionedCall�
flatten_9/PartitionedCallPartitionedCall#dropout_36/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_17266562
flatten_9/PartitionedCall�
 dense_36/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_36_1726676dense_36_1726678*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_36_layer_call_and_return_conditional_losses_17266752"
 dense_36/StatefulPartitionedCall�
dropout_37/PartitionedCallPartitionedCall)dense_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_17266862
dropout_37/PartitionedCall�
 dense_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_37_1726706dense_37_1726708*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_37_layer_call_and_return_conditional_losses_17267052"
 dense_37/StatefulPartitionedCall�
dropout_38/PartitionedCallPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_17267162
dropout_38/PartitionedCall�
 dense_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_38_1726736dense_38_1726738*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_17267352"
 dense_38/StatefulPartitionedCall�
dropout_39/PartitionedCallPartitionedCall)dense_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_17267462
dropout_39/PartitionedCall�
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_36_1726583*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square�
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const�
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/Sum�
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_36/kernel/Regularizer/mul/x�
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul�
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_1726676* 
_output_shapes
:
�@�*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp�
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�@�2$
"dense_36/kernel/Regularizer/Square�
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_36/kernel/Regularizer/Const�
dense_36/kernel/Regularizer/SumSum&dense_36/kernel/Regularizer/Square:y:0*dense_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum�
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_36/kernel/Regularizer/mul/x�
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul�
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_1726706* 
_output_shapes
:
��*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp�
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_37/kernel/Regularizer/Square�
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_37/kernel/Regularizer/Const�
dense_37/kernel/Regularizer/SumSum&dense_37/kernel/Regularizer/Square:y:0*dense_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum�
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_37/kernel/Regularizer/mul/x�
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul�
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_1726736* 
_output_shapes
:
��*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp�
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_38/kernel/Regularizer/Square�
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const�
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum�
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_38/kernel/Regularizer/mul/x�
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul�
IdentityIdentity#dropout_39/PartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall2^dense_36/kernel/Regularizer/Square/ReadVariableOp!^dense_37/StatefulPartitionedCall2^dense_37/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall2^dense_38/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2f
1dense_38/kernel/Regularizer/Square/ReadVariableOp1dense_38/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
��
�)
#__inference__traced_restore_1730265
file_prefix3
 assignvariableop_dense_39_kernel:	�.
 assignvariableop_1_dense_39_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_9_gamma:;
-assignvariableop_8_batch_normalization_9_beta:B
4assignvariableop_9_batch_normalization_9_moving_mean:G
9assignvariableop_10_batch_normalization_9_moving_variance:>
$assignvariableop_11_conv2d_36_kernel: 0
"assignvariableop_12_conv2d_36_bias: ?
$assignvariableop_13_conv2d_37_kernel: �1
"assignvariableop_14_conv2d_37_bias:	�@
$assignvariableop_15_conv2d_38_kernel:��1
"assignvariableop_16_conv2d_38_bias:	�@
$assignvariableop_17_conv2d_39_kernel:��1
"assignvariableop_18_conv2d_39_bias:	�7
#assignvariableop_19_dense_36_kernel:
�@�0
!assignvariableop_20_dense_36_bias:	�7
#assignvariableop_21_dense_37_kernel:
��0
!assignvariableop_22_dense_37_bias:	�7
#assignvariableop_23_dense_38_kernel:
��0
!assignvariableop_24_dense_38_bias:	�#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: =
*assignvariableop_29_adam_dense_39_kernel_m:	�6
(assignvariableop_30_adam_dense_39_bias_m:D
6assignvariableop_31_adam_batch_normalization_9_gamma_m:C
5assignvariableop_32_adam_batch_normalization_9_beta_m:E
+assignvariableop_33_adam_conv2d_36_kernel_m: 7
)assignvariableop_34_adam_conv2d_36_bias_m: F
+assignvariableop_35_adam_conv2d_37_kernel_m: �8
)assignvariableop_36_adam_conv2d_37_bias_m:	�G
+assignvariableop_37_adam_conv2d_38_kernel_m:��8
)assignvariableop_38_adam_conv2d_38_bias_m:	�G
+assignvariableop_39_adam_conv2d_39_kernel_m:��8
)assignvariableop_40_adam_conv2d_39_bias_m:	�>
*assignvariableop_41_adam_dense_36_kernel_m:
�@�7
(assignvariableop_42_adam_dense_36_bias_m:	�>
*assignvariableop_43_adam_dense_37_kernel_m:
��7
(assignvariableop_44_adam_dense_37_bias_m:	�>
*assignvariableop_45_adam_dense_38_kernel_m:
��7
(assignvariableop_46_adam_dense_38_bias_m:	�=
*assignvariableop_47_adam_dense_39_kernel_v:	�6
(assignvariableop_48_adam_dense_39_bias_v:D
6assignvariableop_49_adam_batch_normalization_9_gamma_v:C
5assignvariableop_50_adam_batch_normalization_9_beta_v:E
+assignvariableop_51_adam_conv2d_36_kernel_v: 7
)assignvariableop_52_adam_conv2d_36_bias_v: F
+assignvariableop_53_adam_conv2d_37_kernel_v: �8
)assignvariableop_54_adam_conv2d_37_bias_v:	�G
+assignvariableop_55_adam_conv2d_38_kernel_v:��8
)assignvariableop_56_adam_conv2d_38_bias_v:	�G
+assignvariableop_57_adam_conv2d_39_kernel_v:��8
)assignvariableop_58_adam_conv2d_39_bias_v:	�>
*assignvariableop_59_adam_dense_36_kernel_v:
�@�7
(assignvariableop_60_adam_dense_36_bias_v:	�>
*assignvariableop_61_adam_dense_37_kernel_v:
��7
(assignvariableop_62_adam_dense_37_bias_v:	�>
*assignvariableop_63_adam_dense_38_kernel_v:
��7
(assignvariableop_64_adam_dense_38_bias_v:	�
identity_66��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*�
value�B�BB)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*�
value�B�BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*P
dtypesF
D2B	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_39_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_39_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_9_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_9_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_9_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_9_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_36_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_36_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_37_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_37_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_38_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_38_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_39_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_39_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_36_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_36_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_37_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_37_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp#assignvariableop_23_dense_38_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_dense_38_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_39_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_39_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_9_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_batch_normalization_9_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_36_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_36_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_37_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_37_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_38_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_38_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_39_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_39_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_36_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_36_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_37_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_37_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_38_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_38_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_39_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_39_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_9_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_9_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_36_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_36_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_37_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_37_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_38_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_38_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_39_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_39_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_36_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_36_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_37_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_37_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_38_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_38_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_649
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_65Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_65�
Identity_66IdentityIdentity_65:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_66"#
identity_66Identity_66:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_64AssignVariableOp_642(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
"__inference__wrapped_model_1726347
input_1
cnn3_1726305:
cnn3_1726307:
cnn3_1726309:
cnn3_1726311:&
cnn3_1726313: 
cnn3_1726315: '
cnn3_1726317: �
cnn3_1726319:	�(
cnn3_1726321:��
cnn3_1726323:	�(
cnn3_1726325:��
cnn3_1726327:	� 
cnn3_1726329:
�@�
cnn3_1726331:	� 
cnn3_1726333:
��
cnn3_1726335:	� 
cnn3_1726337:
��
cnn3_1726339:	�
cnn3_1726341:	�
cnn3_1726343:
identity��CNN3/StatefulPartitionedCall�
CNN3/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn3_1726305cnn3_1726307cnn3_1726309cnn3_1726311cnn3_1726313cnn3_1726315cnn3_1726317cnn3_1726319cnn3_1726321cnn3_1726323cnn3_1726325cnn3_1726327cnn3_1726329cnn3_1726331cnn3_1726333cnn3_1726335cnn3_1726337cnn3_1726339cnn3_1726341cnn3_1726343* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *!
fR
__inference_call_16181162
CNN3/StatefulPartitionedCall�
IdentityIdentity%CNN3/StatefulPartitionedCall:output:0^CNN3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2<
CNN3/StatefulPartitionedCallCNN3/StatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
f
G__inference_dropout_39_layer_call_and_return_conditional_losses_1729788

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
�
G
+__inference_flatten_9_layer_call_fn_1729621

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
:����������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_17266562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_39_layer_call_fn_1726521

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_17265152
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
�
e
G__inference_dropout_36_layer_call_and_return_conditional_losses_1729588

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_9_layer_call_fn_1729239

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_17267732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
&__inference_CNN3_layer_call_fn_1728657
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_17276682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
Ϗ
�
__inference_call_1620557

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_38_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_39_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	�H
4sequential_9_dense_36_matmul_readvariableop_resource:
�@�D
5sequential_9_dense_36_biasadd_readvariableop_resource:	�H
4sequential_9_dense_37_matmul_readvariableop_resource:
��D
5sequential_9_dense_37_biasadd_readvariableop_resource:	�H
4sequential_9_dense_38_matmul_readvariableop_resource:
��D
5sequential_9_dense_38_biasadd_readvariableop_resource:	�:
'dense_39_matmul_readvariableop_resource:	�6
(dense_39_biasadd_readvariableop_resource:
identity��dense_39/BiasAdd/ReadVariableOp�dense_39/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�,sequential_9/conv2d_36/Conv2D/ReadVariableOp�-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�,sequential_9/conv2d_37/Conv2D/ReadVariableOp�-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�,sequential_9/conv2d_38/Conv2D/ReadVariableOp�-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�,sequential_9/conv2d_39/Conv2D/ReadVariableOp�,sequential_9/dense_36/BiasAdd/ReadVariableOp�+sequential_9/dense_36/MatMul/ReadVariableOp�,sequential_9/dense_37/BiasAdd/ReadVariableOp�+sequential_9/dense_37/MatMul/ReadVariableOp�,sequential_9/dense_38/BiasAdd/ReadVariableOp�+sequential_9/dense_38/MatMul/ReadVariableOp�
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack�
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1�
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2�
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice�
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp�
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3�
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOp�
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D�
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_9/conv2d_36/BiasAdd�
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_9/conv2d_36/Relu�
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool�
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOp�
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D�
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_9/conv2d_37/BiasAdd�
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_9/conv2d_37/Relu�
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool�
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOp�
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D�
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_9/conv2d_38/BiasAdd�
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_9/conv2d_38/Relu�
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool�
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOp�
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�		�*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D�
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�		�2 
sequential_9/conv2d_39/BiasAdd�
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*(
_output_shapes
:�		�2
sequential_9/conv2d_39/Relu�
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool�
 sequential_9/dropout_36/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*(
_output_shapes
:��2"
 sequential_9/dropout_36/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_36/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0* 
_output_shapes
:
��@2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp�
sequential_9/dense_36/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_36/MatMul�
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp�
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_36/BiasAdd�
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_36/Relu�
 sequential_9/dropout_37/IdentityIdentity(sequential_9/dense_36/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_37/Identity�
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp�
sequential_9/dense_37/MatMulMatMul)sequential_9/dropout_37/Identity:output:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_37/MatMul�
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp�
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_37/BiasAdd�
sequential_9/dense_37/ReluRelu&sequential_9/dense_37/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_37/Relu�
 sequential_9/dropout_38/IdentityIdentity(sequential_9/dense_37/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_38/Identity�
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp�
sequential_9/dense_38/MatMulMatMul)sequential_9/dropout_38/Identity:output:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_38/MatMul�
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp�
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_38/BiasAdd�
sequential_9/dense_38/ReluRelu&sequential_9/dense_38/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_38/Relu�
 sequential_9/dropout_39/IdentityIdentity(sequential_9/dense_38/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_39/Identity�
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_39/MatMul/ReadVariableOp�
dense_39/MatMulMatMul)sequential_9/dropout_39/Identity:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_39/MatMul�
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp�
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_39/BiasAddt
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_39/Softmax�
IdentityIdentitydense_39/Softmax:softmax:0 ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:�KK: : : : : : : : : : : : : : : : : : : : 2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp-sequential_9/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_36/Conv2D/ReadVariableOp,sequential_9/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp-sequential_9/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_37/Conv2D/ReadVariableOp,sequential_9/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp-sequential_9/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_38/Conv2D/ReadVariableOp,sequential_9/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp-sequential_9/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_39/Conv2D/ReadVariableOp,sequential_9/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
��
�
__inference_call_1620733

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_38_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_39_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	�H
4sequential_9_dense_36_matmul_readvariableop_resource:
�@�D
5sequential_9_dense_36_biasadd_readvariableop_resource:	�H
4sequential_9_dense_37_matmul_readvariableop_resource:
��D
5sequential_9_dense_37_biasadd_readvariableop_resource:	�H
4sequential_9_dense_38_matmul_readvariableop_resource:
��D
5sequential_9_dense_38_biasadd_readvariableop_resource:	�:
'dense_39_matmul_readvariableop_resource:	�6
(dense_39_biasadd_readvariableop_resource:
identity��dense_39/BiasAdd/ReadVariableOp�dense_39/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�,sequential_9/conv2d_36/Conv2D/ReadVariableOp�-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�,sequential_9/conv2d_37/Conv2D/ReadVariableOp�-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�,sequential_9/conv2d_38/Conv2D/ReadVariableOp�-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�,sequential_9/conv2d_39/Conv2D/ReadVariableOp�,sequential_9/dense_36/BiasAdd/ReadVariableOp�+sequential_9/dense_36/MatMul/ReadVariableOp�,sequential_9/dense_37/BiasAdd/ReadVariableOp�+sequential_9/dense_37/MatMul/ReadVariableOp�,sequential_9/dense_38/BiasAdd/ReadVariableOp�+sequential_9/dense_38/MatMul/ReadVariableOp�
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack�
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1�
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2�
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice�
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp�
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3�
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOp�
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D�
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp�
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_36/BiasAdd�
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_36/Relu�
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool�
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOp�
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D�
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp�
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_37/BiasAdd�
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_37/Relu�
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool�
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOp�
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D�
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp�
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_38/BiasAdd�
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_38/Relu�
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool�
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOp�
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D�
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp�
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�2 
sequential_9/conv2d_39/BiasAdd�
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
sequential_9/conv2d_39/Relu�
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool�
 sequential_9/dropout_36/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:����������2"
 sequential_9/dropout_36/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_36/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:����������@2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
�@�*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp�
sequential_9/dense_36/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/MatMul�
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp�
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/BiasAdd�
sequential_9/dense_36/ReluRelu&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_36/Relu�
 sequential_9/dropout_37/IdentityIdentity(sequential_9/dense_36/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_37/Identity�
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp�
sequential_9/dense_37/MatMulMatMul)sequential_9/dropout_37/Identity:output:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/MatMul�
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp�
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/BiasAdd�
sequential_9/dense_37/ReluRelu&sequential_9/dense_37/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_37/Relu�
 sequential_9/dropout_38/IdentityIdentity(sequential_9/dense_37/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_38/Identity�
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp�
sequential_9/dense_38/MatMulMatMul)sequential_9/dropout_38/Identity:output:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/MatMul�
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp�
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/BiasAdd�
sequential_9/dense_38/ReluRelu&sequential_9/dense_38/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_38/Relu�
 sequential_9/dropout_39/IdentityIdentity(sequential_9/dense_38/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_39/Identity�
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_39/MatMul/ReadVariableOp�
dense_39/MatMulMatMul)sequential_9/dropout_39/Identity:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/MatMul�
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp�
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_39/BiasAdd|
dense_39/SoftmaxSoftmaxdense_39/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_39/Softmax�
IdentityIdentitydense_39/Softmax:softmax:0 ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp-sequential_9/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_36/Conv2D/ReadVariableOp,sequential_9/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp-sequential_9/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_37/Conv2D/ReadVariableOp,sequential_9/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp-sequential_9/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_38/Conv2D/ReadVariableOp,sequential_9/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp-sequential_9/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_39/Conv2D/ReadVariableOp,sequential_9/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1729616

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729385

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
�
e
,__inference_dropout_37_layer_call_fn_1729680

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_17268982
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
�
�
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1726636

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
:���������		�*
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
:���������		�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������		�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������		�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1726555

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
�
�
&__inference_CNN3_layer_call_fn_1728522
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�@�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_17274982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
e
,__inference_dropout_39_layer_call_fn_1729798

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_17268322
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
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1726413

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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__
	�call"�	
_tf_keras_model�{"name": "CNN3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN3", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN3"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer_with_weights-4
layer-8
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_sequential�~{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_9_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 43, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_9_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_9_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25}, {"class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 26}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31}, {"class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 32}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}, {"class_name": "Dropout", "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 37}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 40}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 41}, {"class_name": "Dropout", "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 42}]}}}
�

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
%iter

&beta_1

'beta_2
	(decay
)learning_ratem� m�*m�+m�.m�/m�0m�1m�2m�3m�4m�5m�6m�7m�8m�9m�:m�;m�v� v�*v�+v�.v�/v�0v�1v�2v�3v�4v�5v�6v�7v�8v�9v�:v�;v�"
	optimizer
 "
trackable_list_wrapper
�
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17
18
 19"
trackable_list_wrapper
�
*0
+1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
16
 17"
trackable_list_wrapper
�

<layers
regularization_losses
	variables
=metrics
trainable_variables
>non_trainable_variables
?layer_regularization_losses
@layer_metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "lambda_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
�

Eaxis
	*gamma
+beta
,moving_mean
-moving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

.kernel
/bias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 50}}
�


0kernel
1bias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
�
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 52}}
�


2kernel
3bias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
�
^regularization_losses
_	variables
`trainable_variables
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 54}}
�


4kernel
5bias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 55}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 9, 9, 256]}}
�
fregularization_losses
g	variables
htrainable_variables
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 56}}
�
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_36", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 26}
�
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 57}}
�	

6kernel
7bias
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 8192]}}
�
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_37", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 32}
�	

8kernel
9bias
zregularization_losses
{	variables
|trainable_variables
}	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
~regularization_losses
	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_38", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 37}
�	

:kernel
;bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 40}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_39", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 42}
@
�0
�1
�2
�3"
trackable_list_wrapper
�
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17"
trackable_list_wrapper
�
*0
+1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15"
trackable_list_wrapper
�
�layers
regularization_losses
	variables
�metrics
trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_39/kernel
:2dense_39/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
�
�layers
!regularization_losses
"	variables
�metrics
#trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
):'2batch_normalization_9/gamma
(:&2batch_normalization_9/beta
1:/ (2!batch_normalization_9/moving_mean
5:3 (2%batch_normalization_9/moving_variance
*:( 2conv2d_36/kernel
: 2conv2d_36/bias
+:) �2conv2d_37/kernel
:�2conv2d_37/bias
,:*��2conv2d_38/kernel
:�2conv2d_38/bias
,:*��2conv2d_39/kernel
:�2conv2d_39/bias
#:!
�@�2dense_36/kernel
:�2dense_36/bias
#:!
��2dense_37/kernel
:�2dense_37/bias
#:!
��2dense_38/kernel
:�2dense_38/bias
.
0
1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
.
,0
-1"
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
�
�layers
Aregularization_losses
B	variables
�metrics
Ctrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
�
�layers
Fregularization_losses
G	variables
�metrics
Htrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
�
�layers
Jregularization_losses
K	variables
�metrics
Ltrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
Nregularization_losses
O	variables
�metrics
Ptrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
�
�layers
Rregularization_losses
S	variables
�metrics
Ttrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
Vregularization_losses
W	variables
�metrics
Xtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
�
�layers
Zregularization_losses
[	variables
�metrics
\trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
^regularization_losses
_	variables
�metrics
`trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
�layers
bregularization_losses
c	variables
�metrics
dtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
fregularization_losses
g	variables
�metrics
htrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
jregularization_losses
k	variables
�metrics
ltrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
nregularization_losses
o	variables
�metrics
ptrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
�
�layers
rregularization_losses
s	variables
�metrics
ttrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
vregularization_losses
w	variables
�metrics
xtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
�
�layers
zregularization_losses
{	variables
�metrics
|trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
~regularization_losses
	variables
�metrics
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
�
�layers
�regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
�layers
�regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 61}
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
.
,0
-1"
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
(
�0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
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
(
�0"
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
(
�0"
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
':%	�2Adam/dense_39/kernel/m
 :2Adam/dense_39/bias/m
.:,2"Adam/batch_normalization_9/gamma/m
-:+2!Adam/batch_normalization_9/beta/m
/:- 2Adam/conv2d_36/kernel/m
!: 2Adam/conv2d_36/bias/m
0:. �2Adam/conv2d_37/kernel/m
": �2Adam/conv2d_37/bias/m
1:/��2Adam/conv2d_38/kernel/m
": �2Adam/conv2d_38/bias/m
1:/��2Adam/conv2d_39/kernel/m
": �2Adam/conv2d_39/bias/m
(:&
�@�2Adam/dense_36/kernel/m
!:�2Adam/dense_36/bias/m
(:&
��2Adam/dense_37/kernel/m
!:�2Adam/dense_37/bias/m
(:&
��2Adam/dense_38/kernel/m
!:�2Adam/dense_38/bias/m
':%	�2Adam/dense_39/kernel/v
 :2Adam/dense_39/bias/v
.:,2"Adam/batch_normalization_9/gamma/v
-:+2!Adam/batch_normalization_9/beta/v
/:- 2Adam/conv2d_36/kernel/v
!: 2Adam/conv2d_36/bias/v
0:. �2Adam/conv2d_37/kernel/v
": �2Adam/conv2d_37/bias/v
1:/��2Adam/conv2d_38/kernel/v
": �2Adam/conv2d_38/bias/v
1:/��2Adam/conv2d_39/kernel/v
": �2Adam/conv2d_39/bias/v
(:&
�@�2Adam/dense_36/kernel/v
!:�2Adam/dense_36/bias/v
(:&
��2Adam/dense_37/kernel/v
!:�2Adam/dense_37/bias/v
(:&
��2Adam/dense_38/kernel/v
!:�2Adam/dense_38/bias/v
�2�
A__inference_CNN3_layer_call_and_return_conditional_losses_1728085
A__inference_CNN3_layer_call_and_return_conditional_losses_1728225
A__inference_CNN3_layer_call_and_return_conditional_losses_1728337
A__inference_CNN3_layer_call_and_return_conditional_losses_1728477�
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
"__inference__wrapped_model_1726347�
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
&__inference_CNN3_layer_call_fn_1728522
&__inference_CNN3_layer_call_fn_1728567
&__inference_CNN3_layer_call_fn_1728612
&__inference_CNN3_layer_call_fn_1728657�
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
__inference_call_1620557
__inference_call_1620645
__inference_call_1620733�
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_1728786
I__inference_sequential_9_layer_call_and_return_conditional_losses_1728919
I__inference_sequential_9_layer_call_and_return_conditional_losses_1729024
I__inference_sequential_9_layer_call_and_return_conditional_losses_1729157�
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
.__inference_sequential_9_layer_call_fn_1729198
.__inference_sequential_9_layer_call_fn_1729239
.__inference_sequential_9_layer_call_fn_1729280
.__inference_sequential_9_layer_call_fn_1729321�
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
E__inference_dense_39_layer_call_and_return_conditional_losses_1729332�
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
*__inference_dense_39_layer_call_fn_1729341�
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
%__inference_signature_wrapper_1727973input_1"�
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
E__inference_lambda_9_layer_call_and_return_conditional_losses_1729349
E__inference_lambda_9_layer_call_and_return_conditional_losses_1729357�
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
*__inference_lambda_9_layer_call_fn_1729362
*__inference_lambda_9_layer_call_fn_1729367�
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729385
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729403
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729421
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729439�
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
7__inference_batch_normalization_9_layer_call_fn_1729452
7__inference_batch_normalization_9_layer_call_fn_1729465
7__inference_batch_normalization_9_layer_call_fn_1729478
7__inference_batch_normalization_9_layer_call_fn_1729491�
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
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1729514�
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
+__inference_conv2d_36_layer_call_fn_1729523�
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
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1726479�
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
2__inference_max_pooling2d_36_layer_call_fn_1726485�
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
�2�
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1729534�
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
+__inference_conv2d_37_layer_call_fn_1729543�
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
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1726491�
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
2__inference_max_pooling2d_37_layer_call_fn_1726497�
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
�2�
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1729554�
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
+__inference_conv2d_38_layer_call_fn_1729563�
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1726503�
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
2__inference_max_pooling2d_38_layer_call_fn_1726509�
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
�2�
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1729574�
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
+__inference_conv2d_39_layer_call_fn_1729583�
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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1726515�
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
2__inference_max_pooling2d_39_layer_call_fn_1726521�
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
G__inference_dropout_36_layer_call_and_return_conditional_losses_1729588
G__inference_dropout_36_layer_call_and_return_conditional_losses_1729600�
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
,__inference_dropout_36_layer_call_fn_1729605
,__inference_dropout_36_layer_call_fn_1729610�
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
F__inference_flatten_9_layer_call_and_return_conditional_losses_1729616�
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
+__inference_flatten_9_layer_call_fn_1729621�
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
E__inference_dense_36_layer_call_and_return_conditional_losses_1729644�
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
*__inference_dense_36_layer_call_fn_1729653�
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
G__inference_dropout_37_layer_call_and_return_conditional_losses_1729658
G__inference_dropout_37_layer_call_and_return_conditional_losses_1729670�
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
,__inference_dropout_37_layer_call_fn_1729675
,__inference_dropout_37_layer_call_fn_1729680�
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
E__inference_dense_37_layer_call_and_return_conditional_losses_1729703�
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
*__inference_dense_37_layer_call_fn_1729712�
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
G__inference_dropout_38_layer_call_and_return_conditional_losses_1729717
G__inference_dropout_38_layer_call_and_return_conditional_losses_1729729�
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
,__inference_dropout_38_layer_call_fn_1729734
,__inference_dropout_38_layer_call_fn_1729739�
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
E__inference_dense_38_layer_call_and_return_conditional_losses_1729762�
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
*__inference_dense_38_layer_call_fn_1729771�
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
G__inference_dropout_39_layer_call_and_return_conditional_losses_1729776
G__inference_dropout_39_layer_call_and_return_conditional_losses_1729788�
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
,__inference_dropout_39_layer_call_fn_1729793
,__inference_dropout_39_layer_call_fn_1729798�
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
__inference_loss_fn_0_1729809�
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
__inference_loss_fn_1_1729820�
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
__inference_loss_fn_2_1729831�
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
__inference_loss_fn_3_1729842�
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
A__inference_CNN3_layer_call_and_return_conditional_losses_1728085z*+,-./0123456789:; ;�8
1�.
(�%
inputs���������KK
p 
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_1728225z*+,-./0123456789:; ;�8
1�.
(�%
inputs���������KK
p
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_1728337{*+,-./0123456789:; <�9
2�/
)�&
input_1���������KK
p 
� "%�"
�
0���������
� �
A__inference_CNN3_layer_call_and_return_conditional_losses_1728477{*+,-./0123456789:; <�9
2�/
)�&
input_1���������KK
p
� "%�"
�
0���������
� �
&__inference_CNN3_layer_call_fn_1728522n*+,-./0123456789:; <�9
2�/
)�&
input_1���������KK
p 
� "�����������
&__inference_CNN3_layer_call_fn_1728567m*+,-./0123456789:; ;�8
1�.
(�%
inputs���������KK
p 
� "�����������
&__inference_CNN3_layer_call_fn_1728612m*+,-./0123456789:; ;�8
1�.
(�%
inputs���������KK
p
� "�����������
&__inference_CNN3_layer_call_fn_1728657n*+,-./0123456789:; <�9
2�/
)�&
input_1���������KK
p
� "�����������
"__inference__wrapped_model_1726347�*+,-./0123456789:; 8�5
.�+
)�&
input_1���������KK
� "3�0
.
output_1"�
output_1����������
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729385�*+,-M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729403�*+,-M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729421r*+,-;�8
1�.
(�%
inputs���������KK
p 
� "-�*
#� 
0���������KK
� �
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1729439r*+,-;�8
1�.
(�%
inputs���������KK
p
� "-�*
#� 
0���������KK
� �
7__inference_batch_normalization_9_layer_call_fn_1729452�*+,-M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
7__inference_batch_normalization_9_layer_call_fn_1729465�*+,-M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
7__inference_batch_normalization_9_layer_call_fn_1729478e*+,-;�8
1�.
(�%
inputs���������KK
p 
� " ����������KK�
7__inference_batch_normalization_9_layer_call_fn_1729491e*+,-;�8
1�.
(�%
inputs���������KK
p
� " ����������KKy
__inference_call_1620557]*+,-./0123456789:; 3�0
)�&
 �
inputs�KK
p
� "�	�y
__inference_call_1620645]*+,-./0123456789:; 3�0
)�&
 �
inputs�KK
p 
� "�	��
__inference_call_1620733m*+,-./0123456789:; ;�8
1�.
(�%
inputs���������KK
p 
� "�����������
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1729514l./7�4
-�*
(�%
inputs���������KK
� "-�*
#� 
0���������KK 
� �
+__inference_conv2d_36_layer_call_fn_1729523_./7�4
-�*
(�%
inputs���������KK
� " ����������KK �
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1729534m017�4
-�*
(�%
inputs���������%% 
� ".�+
$�!
0���������%%�
� �
+__inference_conv2d_37_layer_call_fn_1729543`017�4
-�*
(�%
inputs���������%% 
� "!����������%%��
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1729554n238�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_conv2d_38_layer_call_fn_1729563a238�5
.�+
)�&
inputs����������
� "!������������
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1729574n458�5
.�+
)�&
inputs���������		�
� ".�+
$�!
0���������		�
� �
+__inference_conv2d_39_layer_call_fn_1729583a458�5
.�+
)�&
inputs���������		�
� "!����������		��
E__inference_dense_36_layer_call_and_return_conditional_losses_1729644^670�-
&�#
!�
inputs����������@
� "&�#
�
0����������
� 
*__inference_dense_36_layer_call_fn_1729653Q670�-
&�#
!�
inputs����������@
� "������������
E__inference_dense_37_layer_call_and_return_conditional_losses_1729703^890�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_37_layer_call_fn_1729712Q890�-
&�#
!�
inputs����������
� "������������
E__inference_dense_38_layer_call_and_return_conditional_losses_1729762^:;0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_38_layer_call_fn_1729771Q:;0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_39_layer_call_and_return_conditional_losses_1729332] 0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_39_layer_call_fn_1729341P 0�-
&�#
!�
inputs����������
� "�����������
G__inference_dropout_36_layer_call_and_return_conditional_losses_1729588n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
G__inference_dropout_36_layer_call_and_return_conditional_losses_1729600n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
,__inference_dropout_36_layer_call_fn_1729605a<�9
2�/
)�&
inputs����������
p 
� "!������������
,__inference_dropout_36_layer_call_fn_1729610a<�9
2�/
)�&
inputs����������
p
� "!������������
G__inference_dropout_37_layer_call_and_return_conditional_losses_1729658^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_37_layer_call_and_return_conditional_losses_1729670^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_37_layer_call_fn_1729675Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_37_layer_call_fn_1729680Q4�1
*�'
!�
inputs����������
p
� "������������
G__inference_dropout_38_layer_call_and_return_conditional_losses_1729717^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_38_layer_call_and_return_conditional_losses_1729729^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_38_layer_call_fn_1729734Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_38_layer_call_fn_1729739Q4�1
*�'
!�
inputs����������
p
� "������������
G__inference_dropout_39_layer_call_and_return_conditional_losses_1729776^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_39_layer_call_and_return_conditional_losses_1729788^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_39_layer_call_fn_1729793Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_39_layer_call_fn_1729798Q4�1
*�'
!�
inputs����������
p
� "������������
F__inference_flatten_9_layer_call_and_return_conditional_losses_1729616b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������@
� �
+__inference_flatten_9_layer_call_fn_1729621U8�5
.�+
)�&
inputs����������
� "�����������@�
E__inference_lambda_9_layer_call_and_return_conditional_losses_1729349p?�<
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
E__inference_lambda_9_layer_call_and_return_conditional_losses_1729357p?�<
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
*__inference_lambda_9_layer_call_fn_1729362c?�<
5�2
(�%
inputs���������KK

 
p 
� " ����������KK�
*__inference_lambda_9_layer_call_fn_1729367c?�<
5�2
(�%
inputs���������KK

 
p
� " ����������KK<
__inference_loss_fn_0_1729809.�

� 
� "� <
__inference_loss_fn_1_17298206�

� 
� "� <
__inference_loss_fn_2_17298318�

� 
� "� <
__inference_loss_fn_3_1729842:�

� 
� "� �
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1726479�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_36_layer_call_fn_1726485�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1726491�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_37_layer_call_fn_1726497�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1726503�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_38_layer_call_fn_1726509�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1726515�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_39_layer_call_fn_1726521�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_9_layer_call_and_return_conditional_losses_1728786}*+,-./0123456789:;?�<
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_1728919}*+,-./0123456789:;?�<
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_1729024�*+,-./0123456789:;G�D
=�:
0�-
lambda_9_input���������KK
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_9_layer_call_and_return_conditional_losses_1729157�*+,-./0123456789:;G�D
=�:
0�-
lambda_9_input���������KK
p

 
� "&�#
�
0����������
� �
.__inference_sequential_9_layer_call_fn_1729198x*+,-./0123456789:;G�D
=�:
0�-
lambda_9_input���������KK
p 

 
� "������������
.__inference_sequential_9_layer_call_fn_1729239p*+,-./0123456789:;?�<
5�2
(�%
inputs���������KK
p 

 
� "������������
.__inference_sequential_9_layer_call_fn_1729280p*+,-./0123456789:;?�<
5�2
(�%
inputs���������KK
p

 
� "������������
.__inference_sequential_9_layer_call_fn_1729321x*+,-./0123456789:;G�D
=�:
0�-
lambda_9_input���������KK
p

 
� "������������
%__inference_signature_wrapper_1727973�*+,-./0123456789:; C�@
� 
9�6
4
input_1)�&
input_1���������KK"3�0
.
output_1"�
output_1���������