�
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
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
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
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma
�
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta
�
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean
�
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance
�
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
�
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
: *
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
: *
dtype0
�
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*!
shared_nameconv2d_19/kernel
~
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*'
_output_shapes
: �*
dtype0
u
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_19/bias
n
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes	
:�*
dtype0
�
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_20/kernel

$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_20/bias
n
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes	
:�*
dtype0
}
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���* 
shared_namedense_18/kernel
v
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*!
_output_shapes
:���*
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
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_6/gamma/m
�
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
:*
dtype0
�
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_6/beta/m
�
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_18/kernel/m
�
+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_18/bias/m
{
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_19/kernel/m
�
+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_19/bias/m
|
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_20/kernel/m
�
+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_20/bias/m
|
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*'
shared_nameAdam/dense_18/kernel/m
�
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*!
_output_shapes
:���*
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
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_6/gamma/v
�
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
:*
dtype0
�
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_6/beta/v
�
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_18/kernel/v
�
+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_18/bias/v
{
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_19/kernel/v
�
+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_19/bias/v
|
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_20/kernel/v
�
+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_20/bias/v
|
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*'
shared_nameAdam/dense_18/kernel/v
�
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*!
_output_shapes
:���*
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
�]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�]
value�]B�] B�]
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
�
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
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem�m�&m�'m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�v�v�&v�'v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�
 
v
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
14
15
f
&0
'1
*2
+3
,4
-5
.6
/7
08
19
210
311
12
13
�
regularization_losses
4non_trainable_variables
	variables
5layer_regularization_losses
6metrics

7layers
8layer_metrics
trainable_variables
 
R
9regularization_losses
:	variables
;trainable_variables
<	keras_api
�
=axis
	&gamma
'beta
(moving_mean
)moving_variance
>regularization_losses
?	variables
@trainable_variables
A	keras_api
h

*kernel
+bias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

,kernel
-bias
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

.kernel
/bias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
R
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
R
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

0kernel
1bias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
R
fregularization_losses
g	variables
htrainable_variables
i	keras_api
h

2kernel
3bias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
R
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
 
f
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
V
&0
'1
*2
+3
,4
-5
.6
/7
08
19
210
311
�
regularization_losses
rnon_trainable_variables
	variables
slayer_regularization_losses
tmetrics

ulayers
vlayer_metrics
trainable_variables
NL
VARIABLE_VALUEdense_20/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_20/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
wnon_trainable_variables
regularization_losses
xlayer_regularization_losses
	variables
ymetrics

zlayers
{layer_metrics
trainable_variables
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
VARIABLE_VALUEbatch_normalization_6/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_6/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_6/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_6/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_18/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_18/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_19/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_19/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_20/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_20/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_18/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_18/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_19/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_19/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

|0
}1

0
1
 
 
 
 
�
~non_trainable_variables
9regularization_losses
layer_regularization_losses
:	variables
�metrics
�layers
�layer_metrics
;trainable_variables
 
 

&0
'1
(2
)3

&0
'1
�
�non_trainable_variables
>regularization_losses
 �layer_regularization_losses
?	variables
�metrics
�layers
�layer_metrics
@trainable_variables
 

*0
+1

*0
+1
�
�non_trainable_variables
Bregularization_losses
 �layer_regularization_losses
C	variables
�metrics
�layers
�layer_metrics
Dtrainable_variables
 
 
 
�
�non_trainable_variables
Fregularization_losses
 �layer_regularization_losses
G	variables
�metrics
�layers
�layer_metrics
Htrainable_variables
 

,0
-1

,0
-1
�
�non_trainable_variables
Jregularization_losses
 �layer_regularization_losses
K	variables
�metrics
�layers
�layer_metrics
Ltrainable_variables
 
 
 
�
�non_trainable_variables
Nregularization_losses
 �layer_regularization_losses
O	variables
�metrics
�layers
�layer_metrics
Ptrainable_variables
 

.0
/1

.0
/1
�
�non_trainable_variables
Rregularization_losses
 �layer_regularization_losses
S	variables
�metrics
�layers
�layer_metrics
Ttrainable_variables
 
 
 
�
�non_trainable_variables
Vregularization_losses
 �layer_regularization_losses
W	variables
�metrics
�layers
�layer_metrics
Xtrainable_variables
 
 
 
�
�non_trainable_variables
Zregularization_losses
 �layer_regularization_losses
[	variables
�metrics
�layers
�layer_metrics
\trainable_variables
 
 
 
�
�non_trainable_variables
^regularization_losses
 �layer_regularization_losses
_	variables
�metrics
�layers
�layer_metrics
`trainable_variables
 

00
11

00
11
�
�non_trainable_variables
bregularization_losses
 �layer_regularization_losses
c	variables
�metrics
�layers
�layer_metrics
dtrainable_variables
 
 
 
�
�non_trainable_variables
fregularization_losses
 �layer_regularization_losses
g	variables
�metrics
�layers
�layer_metrics
htrainable_variables
 

20
31

20
31
�
�non_trainable_variables
jregularization_losses
 �layer_regularization_losses
k	variables
�metrics
�layers
�layer_metrics
ltrainable_variables
 
 
 
�
�non_trainable_variables
nregularization_losses
 �layer_regularization_losses
o	variables
�metrics
�layers
�layer_metrics
ptrainable_variables

(0
)1
 
 
f
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
(0
)1
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
VARIABLE_VALUEAdam/dense_20/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_20/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_18/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_18/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_19/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_19/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_20/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_20/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_18/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_18/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_19/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_19/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_20/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_20/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_18/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_18/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_19/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_19/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_20/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_20/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_18/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_18/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_19/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_19/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������KK*
dtype0*$
shape:���������KK
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_1020278
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOpConst*B
Tin;
927	*
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
 __inference__traced_save_1021965
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biastotalcounttotal_1count_1Adam/dense_20/kernel/mAdam/dense_20/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/mAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/mAdam/dense_20/kernel/vAdam/dense_20/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/vAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/v*A
Tin:
826*
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
#__inference__traced_restore_1022134��
�
e
G__inference_dropout_20_layer_call_and_return_conditional_losses_1019288

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
�
.__inference_sequential_6_layer_call_fn_1021319

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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_10196272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_1021388

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
�
�
F__inference_conv2d_18_layer_call_and_return_conditional_losses_1021545

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
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
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_1019866

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
�
e
,__inference_dropout_18_layer_call_fn_1021621

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
:���������		�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_10194322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
e
G__inference_dropout_20_layer_call_and_return_conditional_losses_1021728

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
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021033

inputs;
-batch_normalization_6_readvariableop_resource:=
/batch_normalization_6_readvariableop_1_resource:L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_18_conv2d_readvariableop_resource: 7
)conv2d_18_biasadd_readvariableop_resource: C
(conv2d_19_conv2d_readvariableop_resource: �8
)conv2d_19_biasadd_readvariableop_resource:	�D
(conv2d_20_conv2d_readvariableop_resource:��8
)conv2d_20_biasadd_readvariableop_resource:	�<
'dense_18_matmul_readvariableop_resource:���7
(dense_18_biasadd_readvariableop_resource:	�;
'dense_19_matmul_readvariableop_resource:
��7
(dense_19_biasadd_readvariableop_resource:	�
identity��$batch_normalization_6/AssignNewValue�&batch_normalization_6/AssignNewValue_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
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
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3lambda_6/strided_slice:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_6/FusedBatchNormV3�
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue�
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_18/Conv2D�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_18/BiasAdd~
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_18/Relu�
max_pooling2d_18/MaxPoolMaxPoolconv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_18/MaxPool�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_19/Conv2D�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/Relu�
max_pooling2d_19/MaxPoolMaxPoolconv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_19/MaxPool�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_20/Conv2D/ReadVariableOp�
conv2d_20/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_20/Conv2D�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_20/BiasAdd
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_20/Relu�
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPooly
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_18/dropout/Const�
dropout_18/dropout/MulMul!max_pooling2d_20/MaxPool:output:0!dropout_18/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2
dropout_18/dropout/Mul�
dropout_18/dropout/ShapeShape!max_pooling2d_20/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shape�
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype021
/dropout_18/dropout/random_uniform/RandomUniform�
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_18/dropout/GreaterEqual/y�
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2!
dropout_18/dropout/GreaterEqual�
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout_18/dropout/Cast�
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout_18/dropout/Mul_1s
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_6/Const�
flatten_6/ReshapeReshapedropout_18/dropout/Mul_1:z:0flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_6/Reshape�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_19/dropout/Const�
dropout_19/dropout/MulMuldense_18/Relu:activations:0!dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_19/dropout/Mul
dropout_19/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shape�
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_19/dropout/random_uniform/RandomUniform�
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_19/dropout/GreaterEqual/y�
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_19/dropout/GreaterEqual�
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_19/dropout/Cast�
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_19/dropout/Mul_1�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_19/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
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
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_20/dropout/Const�
dropout_20/dropout/MulMuldense_19/Relu:activations:0!dropout_20/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_20/dropout/Mul
dropout_20/dropout/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shape�
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_20/dropout/random_uniform/RandomUniform�
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_20/dropout/GreaterEqual/y�
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_20/dropout/GreaterEqual�
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_20/dropout/Cast�
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_20/dropout/Mul_1�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
dense_19/kernel/Regularizer/mul�
IdentityIdentitydropout_20/dropout/Mul_1:z:0%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2B
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
�\
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_1019309

inputs+
batch_normalization_6_1019146:+
batch_normalization_6_1019148:+
batch_normalization_6_1019150:+
batch_normalization_6_1019152:+
conv2d_18_1019173: 
conv2d_18_1019175: ,
conv2d_19_1019191: � 
conv2d_19_1019193:	�-
conv2d_20_1019209:�� 
conv2d_20_1019211:	�%
dense_18_1019248:���
dense_18_1019250:	�$
dense_19_1019278:
��
dense_19_1019280:	�
identity��-batch_normalization_6/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall� dense_18/StatefulPartitionedCall�1dense_18/kernel/Regularizer/Square/ReadVariableOp� dense_19/StatefulPartitionedCall�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
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
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_10191262
lambda_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0batch_normalization_6_1019146batch_normalization_6_1019148batch_normalization_6_1019150batch_normalization_6_1019152*
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10191452/
-batch_normalization_6/StatefulPartitionedCall�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_18_1019173conv2d_18_1019175*
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
F__inference_conv2d_18_layer_call_and_return_conditional_losses_10191722#
!conv2d_18/StatefulPartitionedCall�
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_10190812"
 max_pooling2d_18/PartitionedCall�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0conv2d_19_1019191conv2d_19_1019193*
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
F__inference_conv2d_19_layer_call_and_return_conditional_losses_10191902#
!conv2d_19/StatefulPartitionedCall�
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_10190932"
 max_pooling2d_19/PartitionedCall�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0conv2d_20_1019209conv2d_20_1019211*
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
F__inference_conv2d_20_layer_call_and_return_conditional_losses_10192082#
!conv2d_20/StatefulPartitionedCall�
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_10191052"
 max_pooling2d_20/PartitionedCall�
dropout_18/PartitionedCallPartitionedCall)max_pooling2d_20/PartitionedCall:output:0*
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
GPU2 *0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_10192202
dropout_18/PartitionedCall�
flatten_6/PartitionedCallPartitionedCall#dropout_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_10192282
flatten_6/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_18_1019248dense_18_1019250*
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
E__inference_dense_18_layer_call_and_return_conditional_losses_10192472"
 dense_18/StatefulPartitionedCall�
dropout_19/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
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
G__inference_dropout_19_layer_call_and_return_conditional_losses_10192582
dropout_19/PartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_19_1019278dense_19_1019280*
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
E__inference_dense_19_layer_call_and_return_conditional_losses_10192772"
 dense_19/StatefulPartitionedCall�
dropout_20/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
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
G__inference_dropout_20_layer_call_and_return_conditional_losses_10192882
dropout_20/PartitionedCall�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_1019173*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_1019248*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_1019278* 
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
dense_19/kernel/Regularizer/mul�
IdentityIdentity#dropout_20/PartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�a
�	
I__inference_sequential_6_layer_call_and_return_conditional_losses_1019627

inputs+
batch_normalization_6_1019567:+
batch_normalization_6_1019569:+
batch_normalization_6_1019571:+
batch_normalization_6_1019573:+
conv2d_18_1019576: 
conv2d_18_1019578: ,
conv2d_19_1019582: � 
conv2d_19_1019584:	�-
conv2d_20_1019588:�� 
conv2d_20_1019590:	�%
dense_18_1019596:���
dense_18_1019598:	�$
dense_19_1019602:
��
dense_19_1019604:	�
identity��-batch_normalization_6/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall� dense_18/StatefulPartitionedCall�1dense_18/kernel/Regularizer/Square/ReadVariableOp� dense_19/StatefulPartitionedCall�1dense_19/kernel/Regularizer/Square/ReadVariableOp�"dropout_18/StatefulPartitionedCall�"dropout_19/StatefulPartitionedCall�"dropout_20/StatefulPartitionedCall�
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
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_10195252
lambda_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0batch_normalization_6_1019567batch_normalization_6_1019569batch_normalization_6_1019571batch_normalization_6_1019573*
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10194982/
-batch_normalization_6/StatefulPartitionedCall�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_18_1019576conv2d_18_1019578*
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
F__inference_conv2d_18_layer_call_and_return_conditional_losses_10191722#
!conv2d_18/StatefulPartitionedCall�
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_10190812"
 max_pooling2d_18/PartitionedCall�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0conv2d_19_1019582conv2d_19_1019584*
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
F__inference_conv2d_19_layer_call_and_return_conditional_losses_10191902#
!conv2d_19/StatefulPartitionedCall�
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_10190932"
 max_pooling2d_19/PartitionedCall�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0conv2d_20_1019588conv2d_20_1019590*
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
F__inference_conv2d_20_layer_call_and_return_conditional_losses_10192082#
!conv2d_20/StatefulPartitionedCall�
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_10191052"
 max_pooling2d_20/PartitionedCall�
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0*
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
GPU2 *0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_10194322$
"dropout_18/StatefulPartitionedCall�
flatten_6/PartitionedCallPartitionedCall+dropout_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_10192282
flatten_6/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_18_1019596dense_18_1019598*
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
E__inference_dense_18_layer_call_and_return_conditional_losses_10192472"
 dense_18/StatefulPartitionedCall�
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
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
G__inference_dropout_19_layer_call_and_return_conditional_losses_10193932$
"dropout_19/StatefulPartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_19_1019602dense_19_1019604*
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
E__inference_dense_19_layer_call_and_return_conditional_losses_10192772"
 dense_19/StatefulPartitionedCall�
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
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
G__inference_dropout_20_layer_call_and_return_conditional_losses_10193602$
"dropout_20/StatefulPartitionedCall�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_18_1019576*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_1019596*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_19_1019602* 
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
dense_19/kernel/Regularizer/mul�
IdentityIdentity+dropout_20/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall2^dense_19/kernel/Regularizer/Square/ReadVariableOp#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
f
G__inference_dropout_19_layer_call_and_return_conditional_losses_1019393

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
�s
�
__inference_call_939400

inputsH
:sequential_6_batch_normalization_6_readvariableop_resource:J
<sequential_6_batch_normalization_6_readvariableop_1_resource:Y
Ksequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:[
Msequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_18_conv2d_readvariableop_resource: D
6sequential_6_conv2d_18_biasadd_readvariableop_resource: P
5sequential_6_conv2d_19_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_19_biasadd_readvariableop_resource:	�Q
5sequential_6_conv2d_20_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_20_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:���D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�1sequential_6/batch_normalization_6/ReadVariableOp�3sequential_6/batch_normalization_6/ReadVariableOp_1�-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�,sequential_6/conv2d_18/Conv2D/ReadVariableOp�-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�,sequential_6/conv2d_19/Conv2D/ReadVariableOp�-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�,sequential_6/conv2d_20/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
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
1sequential_6/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_6_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_6/batch_normalization_6/ReadVariableOp�
3sequential_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_6/batch_normalization_6/ReadVariableOp_1�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
3sequential_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:09sequential_6/batch_normalization_6/ReadVariableOp:value:0;sequential_6/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 25
3sequential_6/batch_normalization_6/FusedBatchNormV3�
,sequential_6/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_18/Conv2D/ReadVariableOp�
sequential_6/conv2d_18/Conv2DConv2D7sequential_6/batch_normalization_6/FusedBatchNormV3:y:04sequential_6/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_6/conv2d_18/Conv2D�
-sequential_6/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�
sequential_6/conv2d_18/BiasAddBiasAdd&sequential_6/conv2d_18/Conv2D:output:05sequential_6/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_6/conv2d_18/BiasAdd�
sequential_6/conv2d_18/ReluRelu'sequential_6/conv2d_18/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_6/conv2d_18/Relu�
%sequential_6/max_pooling2d_18/MaxPoolMaxPool)sequential_6/conv2d_18/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_18/MaxPool�
,sequential_6/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_19/Conv2D/ReadVariableOp�
sequential_6/conv2d_19/Conv2DConv2D.sequential_6/max_pooling2d_18/MaxPool:output:04sequential_6/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_6/conv2d_19/Conv2D�
-sequential_6/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�
sequential_6/conv2d_19/BiasAddBiasAdd&sequential_6/conv2d_19/Conv2D:output:05sequential_6/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_6/conv2d_19/BiasAdd�
sequential_6/conv2d_19/ReluRelu'sequential_6/conv2d_19/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_6/conv2d_19/Relu�
%sequential_6/max_pooling2d_19/MaxPoolMaxPool)sequential_6/conv2d_19/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_19/MaxPool�
,sequential_6/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_20/Conv2D/ReadVariableOp�
sequential_6/conv2d_20/Conv2DConv2D.sequential_6/max_pooling2d_19/MaxPool:output:04sequential_6/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_6/conv2d_20/Conv2D�
-sequential_6/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�
sequential_6/conv2d_20/BiasAddBiasAdd&sequential_6/conv2d_20/Conv2D:output:05sequential_6/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_6/conv2d_20/BiasAdd�
sequential_6/conv2d_20/ReluRelu'sequential_6/conv2d_20/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_6/conv2d_20/Relu�
%sequential_6/max_pooling2d_20/MaxPoolMaxPool)sequential_6/conv2d_20/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_20/MaxPool�
 sequential_6/dropout_18/IdentityIdentity.sequential_6/max_pooling2d_20/MaxPool:output:0*
T0*(
_output_shapes
:�		�2"
 sequential_6/dropout_18/Identity�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_18/Identity:output:0%sequential_6/flatten_6/Const:output:0*
T0*!
_output_shapes
:���2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
 sequential_6/dropout_19/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_6/dropout_19/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_19/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
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
 sequential_6/dropout_20/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_6/dropout_20/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_20/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
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
dense_20/Softmax�
IdentityIdentitydense_20/Softmax:softmax:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpC^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_6/batch_normalization_6/ReadVariableOp4^sequential_6/batch_normalization_6/ReadVariableOp_1.^sequential_6/conv2d_18/BiasAdd/ReadVariableOp-^sequential_6/conv2d_18/Conv2D/ReadVariableOp.^sequential_6/conv2d_19/BiasAdd/ReadVariableOp-^sequential_6/conv2d_19/Conv2D/ReadVariableOp.^sequential_6/conv2d_20/BiasAdd/ReadVariableOp-^sequential_6/conv2d_20/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:�KK: : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_6/batch_normalization_6/ReadVariableOp1sequential_6/batch_normalization_6/ReadVariableOp2j
3sequential_6/batch_normalization_6/ReadVariableOp_13sequential_6/batch_normalization_6/ReadVariableOp_12^
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp-sequential_6/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_18/Conv2D/ReadVariableOp,sequential_6/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp-sequential_6/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_19/Conv2D/ReadVariableOp,sequential_6/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp-sequential_6/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_20/Conv2D/ReadVariableOp,sequential_6/conv2d_20/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
G
+__inference_flatten_6_layer_call_fn_1021632

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
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_10192282
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
F__inference_conv2d_19_layer_call_and_return_conditional_losses_1021565

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
�
.__inference_sequential_6_layer_call_fn_1021352
lambda_6_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_10196272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_6_input
��
�
@__inference_CNN_layer_call_and_return_conditional_losses_1020680
input_1H
:sequential_6_batch_normalization_6_readvariableop_resource:J
<sequential_6_batch_normalization_6_readvariableop_1_resource:Y
Ksequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:[
Msequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_18_conv2d_readvariableop_resource: D
6sequential_6_conv2d_18_biasadd_readvariableop_resource: P
5sequential_6_conv2d_19_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_19_biasadd_readvariableop_resource:	�Q
5sequential_6_conv2d_20_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_20_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:���D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�1sequential_6/batch_normalization_6/AssignNewValue�3sequential_6/batch_normalization_6/AssignNewValue_1�Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�1sequential_6/batch_normalization_6/ReadVariableOp�3sequential_6/batch_normalization_6/ReadVariableOp_1�-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�,sequential_6/conv2d_18/Conv2D/ReadVariableOp�-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�,sequential_6/conv2d_19/Conv2D/ReadVariableOp�-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�,sequential_6/conv2d_20/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
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
1sequential_6/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_6_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_6/batch_normalization_6/ReadVariableOp�
3sequential_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_6/batch_normalization_6/ReadVariableOp_1�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
3sequential_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:09sequential_6/batch_normalization_6/ReadVariableOp:value:0;sequential_6/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<25
3sequential_6/batch_normalization_6/FusedBatchNormV3�
1sequential_6/batch_normalization_6/AssignNewValueAssignVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource@sequential_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0C^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_6/batch_normalization_6/AssignNewValue�
3sequential_6/batch_normalization_6/AssignNewValue_1AssignVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceDsequential_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0E^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_6/batch_normalization_6/AssignNewValue_1�
,sequential_6/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_18/Conv2D/ReadVariableOp�
sequential_6/conv2d_18/Conv2DConv2D7sequential_6/batch_normalization_6/FusedBatchNormV3:y:04sequential_6/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_18/Conv2D�
-sequential_6/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�
sequential_6/conv2d_18/BiasAddBiasAdd&sequential_6/conv2d_18/Conv2D:output:05sequential_6/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_18/BiasAdd�
sequential_6/conv2d_18/ReluRelu'sequential_6/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_18/Relu�
%sequential_6/max_pooling2d_18/MaxPoolMaxPool)sequential_6/conv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_18/MaxPool�
,sequential_6/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_19/Conv2D/ReadVariableOp�
sequential_6/conv2d_19/Conv2DConv2D.sequential_6/max_pooling2d_18/MaxPool:output:04sequential_6/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_19/Conv2D�
-sequential_6/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�
sequential_6/conv2d_19/BiasAddBiasAdd&sequential_6/conv2d_19/Conv2D:output:05sequential_6/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_19/BiasAdd�
sequential_6/conv2d_19/ReluRelu'sequential_6/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_19/Relu�
%sequential_6/max_pooling2d_19/MaxPoolMaxPool)sequential_6/conv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_19/MaxPool�
,sequential_6/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_20/Conv2D/ReadVariableOp�
sequential_6/conv2d_20/Conv2DConv2D.sequential_6/max_pooling2d_19/MaxPool:output:04sequential_6/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_6/conv2d_20/Conv2D�
-sequential_6/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�
sequential_6/conv2d_20/BiasAddBiasAdd&sequential_6/conv2d_20/Conv2D:output:05sequential_6/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_6/conv2d_20/BiasAdd�
sequential_6/conv2d_20/ReluRelu'sequential_6/conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_6/conv2d_20/Relu�
%sequential_6/max_pooling2d_20/MaxPoolMaxPool)sequential_6/conv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_20/MaxPool�
%sequential_6/dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2'
%sequential_6/dropout_18/dropout/Const�
#sequential_6/dropout_18/dropout/MulMul.sequential_6/max_pooling2d_20/MaxPool:output:0.sequential_6/dropout_18/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2%
#sequential_6/dropout_18/dropout/Mul�
%sequential_6/dropout_18/dropout/ShapeShape.sequential_6/max_pooling2d_20/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_18/dropout/Shape�
<sequential_6/dropout_18/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_18/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype02>
<sequential_6/dropout_18/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=20
.sequential_6/dropout_18/dropout/GreaterEqual/y�
,sequential_6/dropout_18/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_18/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_18/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2.
,sequential_6/dropout_18/dropout/GreaterEqual�
$sequential_6/dropout_18/dropout/CastCast0sequential_6/dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2&
$sequential_6/dropout_18/dropout/Cast�
%sequential_6/dropout_18/dropout/Mul_1Mul'sequential_6/dropout_18/dropout/Mul:z:0(sequential_6/dropout_18/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2'
%sequential_6/dropout_18/dropout/Mul_1�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_18/dropout/Mul_1:z:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
%sequential_6/dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_6/dropout_19/dropout/Const�
#sequential_6/dropout_19/dropout/MulMul(sequential_6/dense_18/Relu:activations:0.sequential_6/dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_6/dropout_19/dropout/Mul�
%sequential_6/dropout_19/dropout/ShapeShape(sequential_6/dense_18/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_19/dropout/Shape�
<sequential_6/dropout_19/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_6/dropout_19/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_6/dropout_19/dropout/GreaterEqual/y�
,sequential_6/dropout_19/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_19/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_6/dropout_19/dropout/GreaterEqual�
$sequential_6/dropout_19/dropout/CastCast0sequential_6/dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_6/dropout_19/dropout/Cast�
%sequential_6/dropout_19/dropout/Mul_1Mul'sequential_6/dropout_19/dropout/Mul:z:0(sequential_6/dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_6/dropout_19/dropout/Mul_1�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_19/dropout/Mul_1:z:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
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
%sequential_6/dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_6/dropout_20/dropout/Const�
#sequential_6/dropout_20/dropout/MulMul(sequential_6/dense_19/Relu:activations:0.sequential_6/dropout_20/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_6/dropout_20/dropout/Mul�
%sequential_6/dropout_20/dropout/ShapeShape(sequential_6/dense_19/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_20/dropout/Shape�
<sequential_6/dropout_20/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_20/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_6/dropout_20/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_6/dropout_20/dropout/GreaterEqual/y�
,sequential_6/dropout_20/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_20/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_20/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_6/dropout_20/dropout/GreaterEqual�
$sequential_6/dropout_20/dropout/CastCast0sequential_6/dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_6/dropout_20/dropout/Cast�
%sequential_6/dropout_20/dropout/Mul_1Mul'sequential_6/dropout_20/dropout/Mul:z:0(sequential_6/dropout_20/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_6/dropout_20/dropout/Mul_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_20/dropout/Mul_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
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
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
dense_19/kernel/Regularizer/mul�	
IdentityIdentitydense_20/Softmax:softmax:03^conv2d_18/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp2^sequential_6/batch_normalization_6/AssignNewValue4^sequential_6/batch_normalization_6/AssignNewValue_1C^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_6/batch_normalization_6/ReadVariableOp4^sequential_6/batch_normalization_6/ReadVariableOp_1.^sequential_6/conv2d_18/BiasAdd/ReadVariableOp-^sequential_6/conv2d_18/Conv2D/ReadVariableOp.^sequential_6/conv2d_19/BiasAdd/ReadVariableOp-^sequential_6/conv2d_19/Conv2D/ReadVariableOp.^sequential_6/conv2d_20/BiasAdd/ReadVariableOp-^sequential_6/conv2d_20/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2f
1sequential_6/batch_normalization_6/AssignNewValue1sequential_6/batch_normalization_6/AssignNewValue2j
3sequential_6/batch_normalization_6/AssignNewValue_13sequential_6/batch_normalization_6/AssignNewValue_12�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_6/batch_normalization_6/ReadVariableOp1sequential_6/batch_normalization_6/ReadVariableOp2j
3sequential_6/batch_normalization_6/ReadVariableOp_13sequential_6/batch_normalization_6/ReadVariableOp_12^
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp-sequential_6/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_18/Conv2D/ReadVariableOp,sequential_6/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp-sequential_6/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_19/Conv2D/ReadVariableOp,sequential_6/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp-sequential_6/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_20/Conv2D/ReadVariableOp,sequential_6/conv2d_20/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
+__inference_conv2d_19_layer_call_fn_1021574

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
F__inference_conv2d_19_layer_call_and_return_conditional_losses_10191902
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
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021434

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
�v
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021116
lambda_6_input;
-batch_normalization_6_readvariableop_resource:=
/batch_normalization_6_readvariableop_1_resource:L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_18_conv2d_readvariableop_resource: 7
)conv2d_18_biasadd_readvariableop_resource: C
(conv2d_19_conv2d_readvariableop_resource: �8
)conv2d_19_biasadd_readvariableop_resource:	�D
(conv2d_20_conv2d_readvariableop_resource:��8
)conv2d_20_biasadd_readvariableop_resource:	�<
'dense_18_matmul_readvariableop_resource:���7
(dense_18_biasadd_readvariableop_resource:	�;
'dense_19_matmul_readvariableop_resource:
��7
(dense_19_biasadd_readvariableop_resource:	�
identity��5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
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
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3lambda_6/strided_slice:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_18/Conv2D�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_18/BiasAdd~
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_18/Relu�
max_pooling2d_18/MaxPoolMaxPoolconv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_18/MaxPool�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_19/Conv2D�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/Relu�
max_pooling2d_19/MaxPoolMaxPoolconv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_19/MaxPool�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_20/Conv2D/ReadVariableOp�
conv2d_20/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_20/Conv2D�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_20/BiasAdd
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_20/Relu�
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool�
dropout_18/IdentityIdentity!max_pooling2d_20/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2
dropout_18/Identitys
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_6/Const�
flatten_6/ReshapeReshapedropout_18/Identity:output:0flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_6/Reshape�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
dropout_19/IdentityIdentitydense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_19/Identity�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_19/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
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
dropout_20/IdentityIdentitydense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_20/Identity�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
dense_19/kernel/Regularizer/mul�
IdentityIdentitydropout_20/Identity:output:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2B
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
�
�
+__inference_conv2d_20_layer_call_fn_1021594

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
F__inference_conv2d_20_layer_call_and_return_conditional_losses_10192082
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
��
�!
#__inference__traced_restore_1022134
file_prefix3
 assignvariableop_dense_20_kernel:	�.
 assignvariableop_1_dense_20_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_6_gamma:;
-assignvariableop_8_batch_normalization_6_beta:B
4assignvariableop_9_batch_normalization_6_moving_mean:G
9assignvariableop_10_batch_normalization_6_moving_variance:>
$assignvariableop_11_conv2d_18_kernel: 0
"assignvariableop_12_conv2d_18_bias: ?
$assignvariableop_13_conv2d_19_kernel: �1
"assignvariableop_14_conv2d_19_bias:	�@
$assignvariableop_15_conv2d_20_kernel:��1
"assignvariableop_16_conv2d_20_bias:	�8
#assignvariableop_17_dense_18_kernel:���0
!assignvariableop_18_dense_18_bias:	�7
#assignvariableop_19_dense_19_kernel:
��0
!assignvariableop_20_dense_19_bias:	�#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_20_kernel_m:	�6
(assignvariableop_26_adam_dense_20_bias_m:D
6assignvariableop_27_adam_batch_normalization_6_gamma_m:C
5assignvariableop_28_adam_batch_normalization_6_beta_m:E
+assignvariableop_29_adam_conv2d_18_kernel_m: 7
)assignvariableop_30_adam_conv2d_18_bias_m: F
+assignvariableop_31_adam_conv2d_19_kernel_m: �8
)assignvariableop_32_adam_conv2d_19_bias_m:	�G
+assignvariableop_33_adam_conv2d_20_kernel_m:��8
)assignvariableop_34_adam_conv2d_20_bias_m:	�?
*assignvariableop_35_adam_dense_18_kernel_m:���7
(assignvariableop_36_adam_dense_18_bias_m:	�>
*assignvariableop_37_adam_dense_19_kernel_m:
��7
(assignvariableop_38_adam_dense_19_bias_m:	�=
*assignvariableop_39_adam_dense_20_kernel_v:	�6
(assignvariableop_40_adam_dense_20_bias_v:D
6assignvariableop_41_adam_batch_normalization_6_gamma_v:C
5assignvariableop_42_adam_batch_normalization_6_beta_v:E
+assignvariableop_43_adam_conv2d_18_kernel_v: 7
)assignvariableop_44_adam_conv2d_18_bias_v: F
+assignvariableop_45_adam_conv2d_19_kernel_v: �8
)assignvariableop_46_adam_conv2d_19_bias_v:	�G
+assignvariableop_47_adam_conv2d_20_kernel_v:��8
)assignvariableop_48_adam_conv2d_20_bias_v:	�?
*assignvariableop_49_adam_dense_18_kernel_v:���7
(assignvariableop_50_adam_dense_18_bias_v:	�>
*assignvariableop_51_adam_dense_19_kernel_v:
��7
(assignvariableop_52_adam_dense_19_bias_v:	�
identity_54��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*�
value�B�6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_6_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_6_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_6_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_6_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_18_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_18_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_19_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_19_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_20_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_20_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_18_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_18_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_19_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_19_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_20_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_20_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_6_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_6_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_18_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_18_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_19_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_19_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_20_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_20_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_18_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_18_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_19_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_19_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_20_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_20_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_6_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_6_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_18_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_18_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_19_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_19_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_20_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_20_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_18_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_18_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_19_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_19_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53�	
Identity_54IdentityIdentity_53:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_54"#
identity_54Identity_54:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
F
*__inference_lambda_6_layer_call_fn_1021398

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
E__inference_lambda_6_layer_call_and_return_conditional_losses_10195252
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
*__inference_dense_19_layer_call_fn_1021723

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
E__inference_dense_19_layer_call_and_return_conditional_losses_10192772
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
�
�
.__inference_sequential_6_layer_call_fn_1021286

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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_10193092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_18_layer_call_fn_1019087

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
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_10190812
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
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_1021627

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_1021761U
;conv2d_18_kernel_regularizer_square_readvariableop_resource: 
identity��2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_18_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
IdentityIdentity$conv2d_18/kernel/Regularizer/mul:z:03^conv2d_18/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp
�
�
F__inference_conv2d_18_layer_call_and_return_conditional_losses_1019172

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
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
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�v
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_1020929

inputs;
-batch_normalization_6_readvariableop_resource:=
/batch_normalization_6_readvariableop_1_resource:L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_18_conv2d_readvariableop_resource: 7
)conv2d_18_biasadd_readvariableop_resource: C
(conv2d_19_conv2d_readvariableop_resource: �8
)conv2d_19_biasadd_readvariableop_resource:	�D
(conv2d_20_conv2d_readvariableop_resource:��8
)conv2d_20_biasadd_readvariableop_resource:	�<
'dense_18_matmul_readvariableop_resource:���7
(dense_18_biasadd_readvariableop_resource:	�;
'dense_19_matmul_readvariableop_resource:
��7
(dense_19_biasadd_readvariableop_resource:	�
identity��5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
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
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3lambda_6/strided_slice:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_18/Conv2D�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_18/BiasAdd~
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_18/Relu�
max_pooling2d_18/MaxPoolMaxPoolconv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_18/MaxPool�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_19/Conv2D�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/Relu�
max_pooling2d_19/MaxPoolMaxPoolconv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_19/MaxPool�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_20/Conv2D/ReadVariableOp�
conv2d_20/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_20/Conv2D�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_20/BiasAdd
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_20/Relu�
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool�
dropout_18/IdentityIdentity!max_pooling2d_20/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2
dropout_18/Identitys
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_6/Const�
flatten_6/ReshapeReshapedropout_18/Identity:output:0flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_6/Reshape�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
dropout_19/IdentityIdentitydense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_19/Identity�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_19/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
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
dropout_20/IdentityIdentitydense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_20/Identity�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
dense_19/kernel/Regularizer/mul�
IdentityIdentitydropout_20/Identity:output:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2B
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
�
�
7__inference_batch_normalization_6_layer_call_fn_1021483

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10189712
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
,__inference_dropout_19_layer_call_fn_1021691

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
G__inference_dropout_19_layer_call_and_return_conditional_losses_10193932
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
�
E__inference_dense_19_layer_call_and_return_conditional_losses_1021714

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
,__inference_dropout_18_layer_call_fn_1021616

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
:���������		�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_10192202
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
.__inference_sequential_6_layer_call_fn_1021253
lambda_6_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: �
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_10193092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_6_input
�
i
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1019105

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
��
�
@__inference_CNN_layer_call_and_return_conditional_losses_1020569
input_1H
:sequential_6_batch_normalization_6_readvariableop_resource:J
<sequential_6_batch_normalization_6_readvariableop_1_resource:Y
Ksequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:[
Msequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_18_conv2d_readvariableop_resource: D
6sequential_6_conv2d_18_biasadd_readvariableop_resource: P
5sequential_6_conv2d_19_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_19_biasadd_readvariableop_resource:	�Q
5sequential_6_conv2d_20_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_20_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:���D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�1sequential_6/batch_normalization_6/ReadVariableOp�3sequential_6/batch_normalization_6/ReadVariableOp_1�-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�,sequential_6/conv2d_18/Conv2D/ReadVariableOp�-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�,sequential_6/conv2d_19/Conv2D/ReadVariableOp�-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�,sequential_6/conv2d_20/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
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
1sequential_6/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_6_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_6/batch_normalization_6/ReadVariableOp�
3sequential_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_6/batch_normalization_6/ReadVariableOp_1�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
3sequential_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:09sequential_6/batch_normalization_6/ReadVariableOp:value:0;sequential_6/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_6/batch_normalization_6/FusedBatchNormV3�
,sequential_6/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_18/Conv2D/ReadVariableOp�
sequential_6/conv2d_18/Conv2DConv2D7sequential_6/batch_normalization_6/FusedBatchNormV3:y:04sequential_6/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_18/Conv2D�
-sequential_6/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�
sequential_6/conv2d_18/BiasAddBiasAdd&sequential_6/conv2d_18/Conv2D:output:05sequential_6/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_18/BiasAdd�
sequential_6/conv2d_18/ReluRelu'sequential_6/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_18/Relu�
%sequential_6/max_pooling2d_18/MaxPoolMaxPool)sequential_6/conv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_18/MaxPool�
,sequential_6/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_19/Conv2D/ReadVariableOp�
sequential_6/conv2d_19/Conv2DConv2D.sequential_6/max_pooling2d_18/MaxPool:output:04sequential_6/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_19/Conv2D�
-sequential_6/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�
sequential_6/conv2d_19/BiasAddBiasAdd&sequential_6/conv2d_19/Conv2D:output:05sequential_6/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_19/BiasAdd�
sequential_6/conv2d_19/ReluRelu'sequential_6/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_19/Relu�
%sequential_6/max_pooling2d_19/MaxPoolMaxPool)sequential_6/conv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_19/MaxPool�
,sequential_6/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_20/Conv2D/ReadVariableOp�
sequential_6/conv2d_20/Conv2DConv2D.sequential_6/max_pooling2d_19/MaxPool:output:04sequential_6/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_6/conv2d_20/Conv2D�
-sequential_6/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�
sequential_6/conv2d_20/BiasAddBiasAdd&sequential_6/conv2d_20/Conv2D:output:05sequential_6/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_6/conv2d_20/BiasAdd�
sequential_6/conv2d_20/ReluRelu'sequential_6/conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_6/conv2d_20/Relu�
%sequential_6/max_pooling2d_20/MaxPoolMaxPool)sequential_6/conv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_20/MaxPool�
 sequential_6/dropout_18/IdentityIdentity.sequential_6/max_pooling2d_20/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_6/dropout_18/Identity�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_18/Identity:output:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
 sequential_6/dropout_19/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_19/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_19/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
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
 sequential_6/dropout_20/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_20/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_20/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
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
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
dense_19/kernel/Regularizer/mul�
IdentityIdentitydense_20/Softmax:softmax:03^conv2d_18/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpC^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_6/batch_normalization_6/ReadVariableOp4^sequential_6/batch_normalization_6/ReadVariableOp_1.^sequential_6/conv2d_18/BiasAdd/ReadVariableOp-^sequential_6/conv2d_18/Conv2D/ReadVariableOp.^sequential_6/conv2d_19/BiasAdd/ReadVariableOp-^sequential_6/conv2d_19/Conv2D/ReadVariableOp.^sequential_6/conv2d_20/BiasAdd/ReadVariableOp-^sequential_6/conv2d_20/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_6/batch_normalization_6/ReadVariableOp1sequential_6/batch_normalization_6/ReadVariableOp2j
3sequential_6/batch_normalization_6/ReadVariableOp_13sequential_6/batch_normalization_6/ReadVariableOp_12^
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp-sequential_6/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_18/Conv2D/ReadVariableOp,sequential_6/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp-sequential_6/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_19/Conv2D/ReadVariableOp,sequential_6/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp-sequential_6/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_20/Conv2D/ReadVariableOp,sequential_6/conv2d_20/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_1021380

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
e
G__inference_dropout_18_layer_call_and_return_conditional_losses_1019220

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:���������		�2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������		�2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
%__inference_CNN_layer_call_fn_1020791

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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_10200312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1019145

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
�
%__inference_signature_wrapper_1020278
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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__wrapped_model_10189492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
��
�
@__inference_CNN_layer_call_and_return_conditional_losses_1020368

inputsH
:sequential_6_batch_normalization_6_readvariableop_resource:J
<sequential_6_batch_normalization_6_readvariableop_1_resource:Y
Ksequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:[
Msequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_18_conv2d_readvariableop_resource: D
6sequential_6_conv2d_18_biasadd_readvariableop_resource: P
5sequential_6_conv2d_19_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_19_biasadd_readvariableop_resource:	�Q
5sequential_6_conv2d_20_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_20_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:���D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�1sequential_6/batch_normalization_6/ReadVariableOp�3sequential_6/batch_normalization_6/ReadVariableOp_1�-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�,sequential_6/conv2d_18/Conv2D/ReadVariableOp�-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�,sequential_6/conv2d_19/Conv2D/ReadVariableOp�-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�,sequential_6/conv2d_20/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
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
1sequential_6/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_6_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_6/batch_normalization_6/ReadVariableOp�
3sequential_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_6/batch_normalization_6/ReadVariableOp_1�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
3sequential_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:09sequential_6/batch_normalization_6/ReadVariableOp:value:0;sequential_6/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_6/batch_normalization_6/FusedBatchNormV3�
,sequential_6/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_18/Conv2D/ReadVariableOp�
sequential_6/conv2d_18/Conv2DConv2D7sequential_6/batch_normalization_6/FusedBatchNormV3:y:04sequential_6/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_18/Conv2D�
-sequential_6/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�
sequential_6/conv2d_18/BiasAddBiasAdd&sequential_6/conv2d_18/Conv2D:output:05sequential_6/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_18/BiasAdd�
sequential_6/conv2d_18/ReluRelu'sequential_6/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_18/Relu�
%sequential_6/max_pooling2d_18/MaxPoolMaxPool)sequential_6/conv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_18/MaxPool�
,sequential_6/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_19/Conv2D/ReadVariableOp�
sequential_6/conv2d_19/Conv2DConv2D.sequential_6/max_pooling2d_18/MaxPool:output:04sequential_6/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_19/Conv2D�
-sequential_6/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�
sequential_6/conv2d_19/BiasAddBiasAdd&sequential_6/conv2d_19/Conv2D:output:05sequential_6/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_19/BiasAdd�
sequential_6/conv2d_19/ReluRelu'sequential_6/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_19/Relu�
%sequential_6/max_pooling2d_19/MaxPoolMaxPool)sequential_6/conv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_19/MaxPool�
,sequential_6/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_20/Conv2D/ReadVariableOp�
sequential_6/conv2d_20/Conv2DConv2D.sequential_6/max_pooling2d_19/MaxPool:output:04sequential_6/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_6/conv2d_20/Conv2D�
-sequential_6/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�
sequential_6/conv2d_20/BiasAddBiasAdd&sequential_6/conv2d_20/Conv2D:output:05sequential_6/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_6/conv2d_20/BiasAdd�
sequential_6/conv2d_20/ReluRelu'sequential_6/conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_6/conv2d_20/Relu�
%sequential_6/max_pooling2d_20/MaxPoolMaxPool)sequential_6/conv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_20/MaxPool�
 sequential_6/dropout_18/IdentityIdentity.sequential_6/max_pooling2d_20/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_6/dropout_18/Identity�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_18/Identity:output:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
 sequential_6/dropout_19/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_19/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_19/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
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
 sequential_6/dropout_20/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_20/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_20/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
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
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
dense_19/kernel/Regularizer/mul�
IdentityIdentitydense_20/Softmax:softmax:03^conv2d_18/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpC^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_6/batch_normalization_6/ReadVariableOp4^sequential_6/batch_normalization_6/ReadVariableOp_1.^sequential_6/conv2d_18/BiasAdd/ReadVariableOp-^sequential_6/conv2d_18/Conv2D/ReadVariableOp.^sequential_6/conv2d_19/BiasAdd/ReadVariableOp-^sequential_6/conv2d_19/Conv2D/ReadVariableOp.^sequential_6/conv2d_20/BiasAdd/ReadVariableOp-^sequential_6/conv2d_20/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_6/batch_normalization_6/ReadVariableOp1sequential_6/batch_normalization_6/ReadVariableOp2j
3sequential_6/batch_normalization_6/ReadVariableOp_13sequential_6/batch_normalization_6/ReadVariableOp_12^
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp-sequential_6/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_18/Conv2D/ReadVariableOp,sequential_6/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp-sequential_6/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_19/Conv2D/ReadVariableOp,sequential_6/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp-sequential_6/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_20/Conv2D/ReadVariableOp,sequential_6/conv2d_20/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1018971

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
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021470

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
�
�
+__inference_conv2d_18_layer_call_fn_1021554

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
F__inference_conv2d_18_layer_call_and_return_conditional_losses_10191722
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
�
e
G__inference_dropout_18_layer_call_and_return_conditional_losses_1021599

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:���������		�2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������		�2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1019498

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
�
�
F__inference_conv2d_20_layer_call_and_return_conditional_losses_1019208

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
�
e
G__inference_dropout_19_layer_call_and_return_conditional_losses_1019258

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
�
N
2__inference_max_pooling2d_20_layer_call_fn_1019111

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
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_10191052
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
�
E__inference_dense_19_layer_call_and_return_conditional_losses_1019277

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
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1019015

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
�
e
G__inference_dropout_19_layer_call_and_return_conditional_losses_1021669

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
__inference_loss_fn_1_1021772O
:dense_18_kernel_regularizer_square_readvariableop_resource:���
identity��1dense_18/kernel/Regularizer/Square/ReadVariableOp�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_18_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
�
i
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_1019081

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
�
�
__inference_loss_fn_2_1021783N
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
�
F
*__inference_lambda_6_layer_call_fn_1021393

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
E__inference_lambda_6_layer_call_and_return_conditional_losses_10191262
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
F__inference_conv2d_19_layer_call_and_return_conditional_losses_1019190

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
�
�
E__inference_dense_18_layer_call_and_return_conditional_losses_1019247

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
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
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_19_layer_call_fn_1021686

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
G__inference_dropout_19_layer_call_and_return_conditional_losses_10192582
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
�
%__inference_CNN_layer_call_fn_1020828
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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_10200312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�2
�
@__inference_CNN_layer_call_and_return_conditional_losses_1020031

inputs"
sequential_6_1019978:"
sequential_6_1019980:"
sequential_6_1019982:"
sequential_6_1019984:.
sequential_6_1019986: "
sequential_6_1019988: /
sequential_6_1019990: �#
sequential_6_1019992:	�0
sequential_6_1019994:��#
sequential_6_1019996:	�)
sequential_6_1019998:���#
sequential_6_1020000:	�(
sequential_6_1020002:
��#
sequential_6_1020004:	�#
dense_20_1020007:	�
dense_20_1020009:
identity��2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp� dense_20/StatefulPartitionedCall�$sequential_6/StatefulPartitionedCall�
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinputssequential_6_1019978sequential_6_1019980sequential_6_1019982sequential_6_1019984sequential_6_1019986sequential_6_1019988sequential_6_1019990sequential_6_1019992sequential_6_1019994sequential_6_1019996sequential_6_1019998sequential_6_1020000sequential_6_1020002sequential_6_1020004*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_10196272&
$sequential_6/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0dense_20_1020007dense_20_1020009*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_10198662"
 dense_20/StatefulPartitionedCall�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_1019986*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_1019998*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_1020002* 
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
IdentityIdentity)dense_20/StatefulPartitionedCall:output:03^conv2d_18/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
f
G__inference_dropout_19_layer_call_and_return_conditional_losses_1021681

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
�2
�
@__inference_CNN_layer_call_and_return_conditional_losses_1019891

inputs"
sequential_6_1019826:"
sequential_6_1019828:"
sequential_6_1019830:"
sequential_6_1019832:.
sequential_6_1019834: "
sequential_6_1019836: /
sequential_6_1019838: �#
sequential_6_1019840:	�0
sequential_6_1019842:��#
sequential_6_1019844:	�)
sequential_6_1019846:���#
sequential_6_1019848:	�(
sequential_6_1019850:
��#
sequential_6_1019852:	�#
dense_20_1019867:	�
dense_20_1019869:
identity��2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp� dense_20/StatefulPartitionedCall�$sequential_6/StatefulPartitionedCall�
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinputssequential_6_1019826sequential_6_1019828sequential_6_1019830sequential_6_1019832sequential_6_1019834sequential_6_1019836sequential_6_1019838sequential_6_1019840sequential_6_1019842sequential_6_1019844sequential_6_1019846sequential_6_1019848sequential_6_1019850sequential_6_1019852*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_10193092&
$sequential_6/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0dense_20_1019867dense_20_1019869*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_10198662"
 dense_20/StatefulPartitionedCall�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_1019834*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_1019846*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
1dense_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_1019850* 
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
IdentityIdentity)dense_20/StatefulPartitionedCall:output:03^conv2d_18/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�h
�
 __inference__traced_save_1021965
file_prefix.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop5
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*�
value�B�6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:: : : : : ::::: : : �:�:��:�:���:�:
��:�: : : : :	�:::: : : �:�:��:�:���:�:
��:�:	�:::: : : �:�:��:�:���:�:
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
:�:'#
!
_output_shapes
:���:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :- )
'
_output_shapes
: �:!!

_output_shapes	
:�:."*
(
_output_shapes
:��:!#

_output_shapes	
:�:'$#
!
_output_shapes
:���:!%

_output_shapes	
:�:&&"
 
_output_shapes
:
��:!'

_output_shapes	
:�:%(!

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
: :-.)
'
_output_shapes
: �:!/

_output_shapes	
:�:.0*
(
_output_shapes
:��:!1

_output_shapes	
:�:'2#
!
_output_shapes
:���:!3

_output_shapes	
:�:&4"
 
_output_shapes
:
��:!5

_output_shapes	
:�:6

_output_shapes
: 
�
e
,__inference_dropout_20_layer_call_fn_1021750

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
G__inference_dropout_20_layer_call_and_return_conditional_losses_10193602
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
7__inference_batch_normalization_6_layer_call_fn_1021522

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10194982
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
�
�
%__inference_CNN_layer_call_fn_1020717
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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_10198912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
*__inference_dense_18_layer_call_fn_1021664

inputs
unknown:���
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
E__inference_dense_18_layer_call_and_return_conditional_losses_10192472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_1019126

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
f
G__inference_dropout_20_layer_call_and_return_conditional_losses_1021740

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
�
"__inference__wrapped_model_1018949
input_1
cnn_1018915:
cnn_1018917:
cnn_1018919:
cnn_1018921:%
cnn_1018923: 
cnn_1018925: &
cnn_1018927: �
cnn_1018929:	�'
cnn_1018931:��
cnn_1018933:	� 
cnn_1018935:���
cnn_1018937:	�
cnn_1018939:
��
cnn_1018941:	�
cnn_1018943:	�
cnn_1018945:
identity��CNN/StatefulPartitionedCall�
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn_1018915cnn_1018917cnn_1018919cnn_1018921cnn_1018923cnn_1018925cnn_1018927cnn_1018929cnn_1018931cnn_1018933cnn_1018935cnn_1018937cnn_1018939cnn_1018941cnn_1018943cnn_1018945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� * 
fR
__inference_call_9373422
CNN/StatefulPartitionedCall�
IdentityIdentity$CNN/StatefulPartitionedCall:output:0^CNN/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2:
CNN/StatefulPartitionedCallCNN/StatefulPartitionedCall:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_1019525

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
�
7__inference_batch_normalization_6_layer_call_fn_1021496

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10190152
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
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_1021363

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
*__inference_dense_20_layer_call_fn_1021372

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
E__inference_dense_20_layer_call_and_return_conditional_losses_10198662
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
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021416

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
�
F__inference_conv2d_20_layer_call_and_return_conditional_losses_1021585

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
�
N
2__inference_max_pooling2d_19_layer_call_fn_1019099

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
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_10190932
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
�
�
%__inference_CNN_layer_call_fn_1020754

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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_10198912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_1019228

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�u
�
__inference_call_937342

inputsH
:sequential_6_batch_normalization_6_readvariableop_resource:J
<sequential_6_batch_normalization_6_readvariableop_1_resource:Y
Ksequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:[
Msequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_18_conv2d_readvariableop_resource: D
6sequential_6_conv2d_18_biasadd_readvariableop_resource: P
5sequential_6_conv2d_19_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_19_biasadd_readvariableop_resource:	�Q
5sequential_6_conv2d_20_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_20_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:���D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�1sequential_6/batch_normalization_6/ReadVariableOp�3sequential_6/batch_normalization_6/ReadVariableOp_1�-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�,sequential_6/conv2d_18/Conv2D/ReadVariableOp�-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�,sequential_6/conv2d_19/Conv2D/ReadVariableOp�-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�,sequential_6/conv2d_20/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
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
1sequential_6/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_6_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_6/batch_normalization_6/ReadVariableOp�
3sequential_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_6/batch_normalization_6/ReadVariableOp_1�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
3sequential_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:09sequential_6/batch_normalization_6/ReadVariableOp:value:0;sequential_6/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_6/batch_normalization_6/FusedBatchNormV3�
,sequential_6/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_18/Conv2D/ReadVariableOp�
sequential_6/conv2d_18/Conv2DConv2D7sequential_6/batch_normalization_6/FusedBatchNormV3:y:04sequential_6/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_18/Conv2D�
-sequential_6/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�
sequential_6/conv2d_18/BiasAddBiasAdd&sequential_6/conv2d_18/Conv2D:output:05sequential_6/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_18/BiasAdd�
sequential_6/conv2d_18/ReluRelu'sequential_6/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_18/Relu�
%sequential_6/max_pooling2d_18/MaxPoolMaxPool)sequential_6/conv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_18/MaxPool�
,sequential_6/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_19/Conv2D/ReadVariableOp�
sequential_6/conv2d_19/Conv2DConv2D.sequential_6/max_pooling2d_18/MaxPool:output:04sequential_6/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_19/Conv2D�
-sequential_6/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�
sequential_6/conv2d_19/BiasAddBiasAdd&sequential_6/conv2d_19/Conv2D:output:05sequential_6/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_19/BiasAdd�
sequential_6/conv2d_19/ReluRelu'sequential_6/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_19/Relu�
%sequential_6/max_pooling2d_19/MaxPoolMaxPool)sequential_6/conv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_19/MaxPool�
,sequential_6/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_20/Conv2D/ReadVariableOp�
sequential_6/conv2d_20/Conv2DConv2D.sequential_6/max_pooling2d_19/MaxPool:output:04sequential_6/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_6/conv2d_20/Conv2D�
-sequential_6/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�
sequential_6/conv2d_20/BiasAddBiasAdd&sequential_6/conv2d_20/Conv2D:output:05sequential_6/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_6/conv2d_20/BiasAdd�
sequential_6/conv2d_20/ReluRelu'sequential_6/conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_6/conv2d_20/Relu�
%sequential_6/max_pooling2d_20/MaxPoolMaxPool)sequential_6/conv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_20/MaxPool�
 sequential_6/dropout_18/IdentityIdentity.sequential_6/max_pooling2d_20/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_6/dropout_18/Identity�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_18/Identity:output:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
 sequential_6/dropout_19/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_19/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_19/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
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
 sequential_6/dropout_20/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_20/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_20/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
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
dense_20/Softmax�
IdentityIdentitydense_20/Softmax:softmax:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpC^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_6/batch_normalization_6/ReadVariableOp4^sequential_6/batch_normalization_6/ReadVariableOp_1.^sequential_6/conv2d_18/BiasAdd/ReadVariableOp-^sequential_6/conv2d_18/Conv2D/ReadVariableOp.^sequential_6/conv2d_19/BiasAdd/ReadVariableOp-^sequential_6/conv2d_19/Conv2D/ReadVariableOp.^sequential_6/conv2d_20/BiasAdd/ReadVariableOp-^sequential_6/conv2d_20/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_6/batch_normalization_6/ReadVariableOp1sequential_6/batch_normalization_6/ReadVariableOp2j
3sequential_6/batch_normalization_6/ReadVariableOp_13sequential_6/batch_normalization_6/ReadVariableOp_12^
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp-sequential_6/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_18/Conv2D/ReadVariableOp,sequential_6/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp-sequential_6/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_19/Conv2D/ReadVariableOp,sequential_6/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp-sequential_6/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_20/Conv2D/ReadVariableOp,sequential_6/conv2d_20/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_1019093

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
�u
�
__inference_call_939472

inputsH
:sequential_6_batch_normalization_6_readvariableop_resource:J
<sequential_6_batch_normalization_6_readvariableop_1_resource:Y
Ksequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:[
Msequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_18_conv2d_readvariableop_resource: D
6sequential_6_conv2d_18_biasadd_readvariableop_resource: P
5sequential_6_conv2d_19_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_19_biasadd_readvariableop_resource:	�Q
5sequential_6_conv2d_20_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_20_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:���D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�1sequential_6/batch_normalization_6/ReadVariableOp�3sequential_6/batch_normalization_6/ReadVariableOp_1�-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�,sequential_6/conv2d_18/Conv2D/ReadVariableOp�-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�,sequential_6/conv2d_19/Conv2D/ReadVariableOp�-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�,sequential_6/conv2d_20/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
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
1sequential_6/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_6_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_6/batch_normalization_6/ReadVariableOp�
3sequential_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_6/batch_normalization_6/ReadVariableOp_1�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
3sequential_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:09sequential_6/batch_normalization_6/ReadVariableOp:value:0;sequential_6/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_6/batch_normalization_6/FusedBatchNormV3�
,sequential_6/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_18/Conv2D/ReadVariableOp�
sequential_6/conv2d_18/Conv2DConv2D7sequential_6/batch_normalization_6/FusedBatchNormV3:y:04sequential_6/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_18/Conv2D�
-sequential_6/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�
sequential_6/conv2d_18/BiasAddBiasAdd&sequential_6/conv2d_18/Conv2D:output:05sequential_6/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_18/BiasAdd�
sequential_6/conv2d_18/ReluRelu'sequential_6/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_18/Relu�
%sequential_6/max_pooling2d_18/MaxPoolMaxPool)sequential_6/conv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_18/MaxPool�
,sequential_6/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_19/Conv2D/ReadVariableOp�
sequential_6/conv2d_19/Conv2DConv2D.sequential_6/max_pooling2d_18/MaxPool:output:04sequential_6/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_19/Conv2D�
-sequential_6/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�
sequential_6/conv2d_19/BiasAddBiasAdd&sequential_6/conv2d_19/Conv2D:output:05sequential_6/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_19/BiasAdd�
sequential_6/conv2d_19/ReluRelu'sequential_6/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_19/Relu�
%sequential_6/max_pooling2d_19/MaxPoolMaxPool)sequential_6/conv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_19/MaxPool�
,sequential_6/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_20/Conv2D/ReadVariableOp�
sequential_6/conv2d_20/Conv2DConv2D.sequential_6/max_pooling2d_19/MaxPool:output:04sequential_6/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_6/conv2d_20/Conv2D�
-sequential_6/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�
sequential_6/conv2d_20/BiasAddBiasAdd&sequential_6/conv2d_20/Conv2D:output:05sequential_6/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_6/conv2d_20/BiasAdd�
sequential_6/conv2d_20/ReluRelu'sequential_6/conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_6/conv2d_20/Relu�
%sequential_6/max_pooling2d_20/MaxPoolMaxPool)sequential_6/conv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_20/MaxPool�
 sequential_6/dropout_18/IdentityIdentity.sequential_6/max_pooling2d_20/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_6/dropout_18/Identity�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_18/Identity:output:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
 sequential_6/dropout_19/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_19/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_19/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
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
 sequential_6/dropout_20/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_6/dropout_20/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_20/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
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
dense_20/Softmax�
IdentityIdentitydense_20/Softmax:softmax:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpC^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_6/batch_normalization_6/ReadVariableOp4^sequential_6/batch_normalization_6/ReadVariableOp_1.^sequential_6/conv2d_18/BiasAdd/ReadVariableOp-^sequential_6/conv2d_18/Conv2D/ReadVariableOp.^sequential_6/conv2d_19/BiasAdd/ReadVariableOp-^sequential_6/conv2d_19/Conv2D/ReadVariableOp.^sequential_6/conv2d_20/BiasAdd/ReadVariableOp-^sequential_6/conv2d_20/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_6/batch_normalization_6/ReadVariableOp1sequential_6/batch_normalization_6/ReadVariableOp2j
3sequential_6/batch_normalization_6/ReadVariableOp_13sequential_6/batch_normalization_6/ReadVariableOp_12^
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp-sequential_6/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_18/Conv2D/ReadVariableOp,sequential_6/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp-sequential_6/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_19/Conv2D/ReadVariableOp,sequential_6/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp-sequential_6/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_20/Conv2D/ReadVariableOp,sequential_6/conv2d_20/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_1021611

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
:���������		�2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
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
:���������		�2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_6_layer_call_fn_1021509

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10191452
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
��
�
@__inference_CNN_layer_call_and_return_conditional_losses_1020479

inputsH
:sequential_6_batch_normalization_6_readvariableop_resource:J
<sequential_6_batch_normalization_6_readvariableop_1_resource:Y
Ksequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:[
Msequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_18_conv2d_readvariableop_resource: D
6sequential_6_conv2d_18_biasadd_readvariableop_resource: P
5sequential_6_conv2d_19_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_19_biasadd_readvariableop_resource:	�Q
5sequential_6_conv2d_20_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_20_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:���D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�1sequential_6/batch_normalization_6/AssignNewValue�3sequential_6/batch_normalization_6/AssignNewValue_1�Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�1sequential_6/batch_normalization_6/ReadVariableOp�3sequential_6/batch_normalization_6/ReadVariableOp_1�-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�,sequential_6/conv2d_18/Conv2D/ReadVariableOp�-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�,sequential_6/conv2d_19/Conv2D/ReadVariableOp�-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�,sequential_6/conv2d_20/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
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
1sequential_6/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_6_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_6/batch_normalization_6/ReadVariableOp�
3sequential_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_6/batch_normalization_6/ReadVariableOp_1�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
3sequential_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:09sequential_6/batch_normalization_6/ReadVariableOp:value:0;sequential_6/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<25
3sequential_6/batch_normalization_6/FusedBatchNormV3�
1sequential_6/batch_normalization_6/AssignNewValueAssignVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource@sequential_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0C^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_6/batch_normalization_6/AssignNewValue�
3sequential_6/batch_normalization_6/AssignNewValue_1AssignVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceDsequential_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0E^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_6/batch_normalization_6/AssignNewValue_1�
,sequential_6/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_18/Conv2D/ReadVariableOp�
sequential_6/conv2d_18/Conv2DConv2D7sequential_6/batch_normalization_6/FusedBatchNormV3:y:04sequential_6/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_6/conv2d_18/Conv2D�
-sequential_6/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�
sequential_6/conv2d_18/BiasAddBiasAdd&sequential_6/conv2d_18/Conv2D:output:05sequential_6/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_6/conv2d_18/BiasAdd�
sequential_6/conv2d_18/ReluRelu'sequential_6/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_6/conv2d_18/Relu�
%sequential_6/max_pooling2d_18/MaxPoolMaxPool)sequential_6/conv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_18/MaxPool�
,sequential_6/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_19/Conv2D/ReadVariableOp�
sequential_6/conv2d_19/Conv2DConv2D.sequential_6/max_pooling2d_18/MaxPool:output:04sequential_6/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_6/conv2d_19/Conv2D�
-sequential_6/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�
sequential_6/conv2d_19/BiasAddBiasAdd&sequential_6/conv2d_19/Conv2D:output:05sequential_6/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_6/conv2d_19/BiasAdd�
sequential_6/conv2d_19/ReluRelu'sequential_6/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_6/conv2d_19/Relu�
%sequential_6/max_pooling2d_19/MaxPoolMaxPool)sequential_6/conv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_19/MaxPool�
,sequential_6/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_20/Conv2D/ReadVariableOp�
sequential_6/conv2d_20/Conv2DConv2D.sequential_6/max_pooling2d_19/MaxPool:output:04sequential_6/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_6/conv2d_20/Conv2D�
-sequential_6/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�
sequential_6/conv2d_20/BiasAddBiasAdd&sequential_6/conv2d_20/Conv2D:output:05sequential_6/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_6/conv2d_20/BiasAdd�
sequential_6/conv2d_20/ReluRelu'sequential_6/conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_6/conv2d_20/Relu�
%sequential_6/max_pooling2d_20/MaxPoolMaxPool)sequential_6/conv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_20/MaxPool�
%sequential_6/dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2'
%sequential_6/dropout_18/dropout/Const�
#sequential_6/dropout_18/dropout/MulMul.sequential_6/max_pooling2d_20/MaxPool:output:0.sequential_6/dropout_18/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2%
#sequential_6/dropout_18/dropout/Mul�
%sequential_6/dropout_18/dropout/ShapeShape.sequential_6/max_pooling2d_20/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_18/dropout/Shape�
<sequential_6/dropout_18/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_18/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype02>
<sequential_6/dropout_18/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=20
.sequential_6/dropout_18/dropout/GreaterEqual/y�
,sequential_6/dropout_18/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_18/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_18/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2.
,sequential_6/dropout_18/dropout/GreaterEqual�
$sequential_6/dropout_18/dropout/CastCast0sequential_6/dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2&
$sequential_6/dropout_18/dropout/Cast�
%sequential_6/dropout_18/dropout/Mul_1Mul'sequential_6/dropout_18/dropout/Mul:z:0(sequential_6/dropout_18/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2'
%sequential_6/dropout_18/dropout/Mul_1�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_18/dropout/Mul_1:z:0%sequential_6/flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
%sequential_6/dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_6/dropout_19/dropout/Const�
#sequential_6/dropout_19/dropout/MulMul(sequential_6/dense_18/Relu:activations:0.sequential_6/dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_6/dropout_19/dropout/Mul�
%sequential_6/dropout_19/dropout/ShapeShape(sequential_6/dense_18/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_19/dropout/Shape�
<sequential_6/dropout_19/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_6/dropout_19/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_6/dropout_19/dropout/GreaterEqual/y�
,sequential_6/dropout_19/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_19/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_6/dropout_19/dropout/GreaterEqual�
$sequential_6/dropout_19/dropout/CastCast0sequential_6/dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_6/dropout_19/dropout/Cast�
%sequential_6/dropout_19/dropout/Mul_1Mul'sequential_6/dropout_19/dropout/Mul:z:0(sequential_6/dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_6/dropout_19/dropout/Mul_1�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_19/dropout/Mul_1:z:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
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
%sequential_6/dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_6/dropout_20/dropout/Const�
#sequential_6/dropout_20/dropout/MulMul(sequential_6/dense_19/Relu:activations:0.sequential_6/dropout_20/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_6/dropout_20/dropout/Mul�
%sequential_6/dropout_20/dropout/ShapeShape(sequential_6/dense_19/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_6/dropout_20/dropout/Shape�
<sequential_6/dropout_20/dropout/random_uniform/RandomUniformRandomUniform.sequential_6/dropout_20/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_6/dropout_20/dropout/random_uniform/RandomUniform�
.sequential_6/dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_6/dropout_20/dropout/GreaterEqual/y�
,sequential_6/dropout_20/dropout/GreaterEqualGreaterEqualEsequential_6/dropout_20/dropout/random_uniform/RandomUniform:output:07sequential_6/dropout_20/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_6/dropout_20/dropout/GreaterEqual�
$sequential_6/dropout_20/dropout/CastCast0sequential_6/dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_6/dropout_20/dropout/Cast�
%sequential_6/dropout_20/dropout/Mul_1Mul'sequential_6/dropout_20/dropout/Mul:z:0(sequential_6/dropout_20/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_6/dropout_20/dropout/Mul_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_20/dropout/Mul_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
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
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
dense_19/kernel/Regularizer/mul�	
IdentityIdentitydense_20/Softmax:softmax:03^conv2d_18/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp2^sequential_6/batch_normalization_6/AssignNewValue4^sequential_6/batch_normalization_6/AssignNewValue_1C^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_6/batch_normalization_6/ReadVariableOp4^sequential_6/batch_normalization_6/ReadVariableOp_1.^sequential_6/conv2d_18/BiasAdd/ReadVariableOp-^sequential_6/conv2d_18/Conv2D/ReadVariableOp.^sequential_6/conv2d_19/BiasAdd/ReadVariableOp-^sequential_6/conv2d_19/Conv2D/ReadVariableOp.^sequential_6/conv2d_20/BiasAdd/ReadVariableOp-^sequential_6/conv2d_20/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2f
1dense_19/kernel/Regularizer/Square/ReadVariableOp1dense_19/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2f
1sequential_6/batch_normalization_6/AssignNewValue1sequential_6/batch_normalization_6/AssignNewValue2j
3sequential_6/batch_normalization_6/AssignNewValue_13sequential_6/batch_normalization_6/AssignNewValue_12�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_6/batch_normalization_6/ReadVariableOp1sequential_6/batch_normalization_6/ReadVariableOp2j
3sequential_6/batch_normalization_6/ReadVariableOp_13sequential_6/batch_normalization_6/ReadVariableOp_12^
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp-sequential_6/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_18/Conv2D/ReadVariableOp,sequential_6/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp-sequential_6/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_19/Conv2D/ReadVariableOp,sequential_6/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp-sequential_6/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_20/Conv2D/ReadVariableOp,sequential_6/conv2d_20/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
H
,__inference_dropout_20_layer_call_fn_1021745

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
G__inference_dropout_20_layer_call_and_return_conditional_losses_10192882
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
�s
�
__inference_call_939328

inputsH
:sequential_6_batch_normalization_6_readvariableop_resource:J
<sequential_6_batch_normalization_6_readvariableop_1_resource:Y
Ksequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:[
Msequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_18_conv2d_readvariableop_resource: D
6sequential_6_conv2d_18_biasadd_readvariableop_resource: P
5sequential_6_conv2d_19_conv2d_readvariableop_resource: �E
6sequential_6_conv2d_19_biasadd_readvariableop_resource:	�Q
5sequential_6_conv2d_20_conv2d_readvariableop_resource:��E
6sequential_6_conv2d_20_biasadd_readvariableop_resource:	�I
4sequential_6_dense_18_matmul_readvariableop_resource:���D
5sequential_6_dense_18_biasadd_readvariableop_resource:	�H
4sequential_6_dense_19_matmul_readvariableop_resource:
��D
5sequential_6_dense_19_biasadd_readvariableop_resource:	�:
'dense_20_matmul_readvariableop_resource:	�6
(dense_20_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�1sequential_6/batch_normalization_6/ReadVariableOp�3sequential_6/batch_normalization_6/ReadVariableOp_1�-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�,sequential_6/conv2d_18/Conv2D/ReadVariableOp�-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�,sequential_6/conv2d_19/Conv2D/ReadVariableOp�-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�,sequential_6/conv2d_20/Conv2D/ReadVariableOp�,sequential_6/dense_18/BiasAdd/ReadVariableOp�+sequential_6/dense_18/MatMul/ReadVariableOp�,sequential_6/dense_19/BiasAdd/ReadVariableOp�+sequential_6/dense_19/MatMul/ReadVariableOp�
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
1sequential_6/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_6_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_6/batch_normalization_6/ReadVariableOp�
3sequential_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_6/batch_normalization_6/ReadVariableOp_1�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
3sequential_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,sequential_6/lambda_6/strided_slice:output:09sequential_6/batch_normalization_6/ReadVariableOp:value:0;sequential_6/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 25
3sequential_6/batch_normalization_6/FusedBatchNormV3�
,sequential_6/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_18/Conv2D/ReadVariableOp�
sequential_6/conv2d_18/Conv2DConv2D7sequential_6/batch_normalization_6/FusedBatchNormV3:y:04sequential_6/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_6/conv2d_18/Conv2D�
-sequential_6/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp�
sequential_6/conv2d_18/BiasAddBiasAdd&sequential_6/conv2d_18/Conv2D:output:05sequential_6/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_6/conv2d_18/BiasAdd�
sequential_6/conv2d_18/ReluRelu'sequential_6/conv2d_18/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_6/conv2d_18/Relu�
%sequential_6/max_pooling2d_18/MaxPoolMaxPool)sequential_6/conv2d_18/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_18/MaxPool�
,sequential_6/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_6/conv2d_19/Conv2D/ReadVariableOp�
sequential_6/conv2d_19/Conv2DConv2D.sequential_6/max_pooling2d_18/MaxPool:output:04sequential_6/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_6/conv2d_19/Conv2D�
-sequential_6/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp�
sequential_6/conv2d_19/BiasAddBiasAdd&sequential_6/conv2d_19/Conv2D:output:05sequential_6/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_6/conv2d_19/BiasAdd�
sequential_6/conv2d_19/ReluRelu'sequential_6/conv2d_19/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_6/conv2d_19/Relu�
%sequential_6/max_pooling2d_19/MaxPoolMaxPool)sequential_6/conv2d_19/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_19/MaxPool�
,sequential_6/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_6/conv2d_20/Conv2D/ReadVariableOp�
sequential_6/conv2d_20/Conv2DConv2D.sequential_6/max_pooling2d_19/MaxPool:output:04sequential_6/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_6/conv2d_20/Conv2D�
-sequential_6/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp�
sequential_6/conv2d_20/BiasAddBiasAdd&sequential_6/conv2d_20/Conv2D:output:05sequential_6/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_6/conv2d_20/BiasAdd�
sequential_6/conv2d_20/ReluRelu'sequential_6/conv2d_20/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_6/conv2d_20/Relu�
%sequential_6/max_pooling2d_20/MaxPoolMaxPool)sequential_6/conv2d_20/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_20/MaxPool�
 sequential_6/dropout_18/IdentityIdentity.sequential_6/max_pooling2d_20/MaxPool:output:0*
T0*(
_output_shapes
:�		�2"
 sequential_6/dropout_18/Identity�
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_6/flatten_6/Const�
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_18/Identity:output:0%sequential_6/flatten_6/Const:output:0*
T0*!
_output_shapes
:���2 
sequential_6/flatten_6/Reshape�
+sequential_6/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
 sequential_6/dropout_19/IdentityIdentity(sequential_6/dense_18/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_6/dropout_19/Identity�
+sequential_6/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_6/dense_19/MatMul/ReadVariableOp�
sequential_6/dense_19/MatMulMatMul)sequential_6/dropout_19/Identity:output:03sequential_6/dense_19/MatMul/ReadVariableOp:value:0*
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
 sequential_6/dropout_20/IdentityIdentity(sequential_6/dense_19/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_6/dropout_20/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)sequential_6/dropout_20/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
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
dense_20/Softmax�
IdentityIdentitydense_20/Softmax:softmax:0 ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOpC^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_6/batch_normalization_6/ReadVariableOp4^sequential_6/batch_normalization_6/ReadVariableOp_1.^sequential_6/conv2d_18/BiasAdd/ReadVariableOp-^sequential_6/conv2d_18/Conv2D/ReadVariableOp.^sequential_6/conv2d_19/BiasAdd/ReadVariableOp-^sequential_6/conv2d_19/Conv2D/ReadVariableOp.^sequential_6/conv2d_20/BiasAdd/ReadVariableOp-^sequential_6/conv2d_20/Conv2D/ReadVariableOp-^sequential_6/dense_18/BiasAdd/ReadVariableOp,^sequential_6/dense_18/MatMul/ReadVariableOp-^sequential_6/dense_19/BiasAdd/ReadVariableOp,^sequential_6/dense_19/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:�KK: : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2�
Bsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_6/batch_normalization_6/ReadVariableOp1sequential_6/batch_normalization_6/ReadVariableOp2j
3sequential_6/batch_normalization_6/ReadVariableOp_13sequential_6/batch_normalization_6/ReadVariableOp_12^
-sequential_6/conv2d_18/BiasAdd/ReadVariableOp-sequential_6/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_18/Conv2D/ReadVariableOp,sequential_6/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_19/BiasAdd/ReadVariableOp-sequential_6/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_19/Conv2D/ReadVariableOp,sequential_6/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_20/BiasAdd/ReadVariableOp-sequential_6/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_20/Conv2D/ReadVariableOp,sequential_6/conv2d_20/Conv2D/ReadVariableOp2\
,sequential_6/dense_18/BiasAdd/ReadVariableOp,sequential_6/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_18/MatMul/ReadVariableOp+sequential_6/dense_18/MatMul/ReadVariableOp2\
,sequential_6/dense_19/BiasAdd/ReadVariableOp,sequential_6/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_19/MatMul/ReadVariableOp+sequential_6/dense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
f
G__inference_dropout_20_layer_call_and_return_conditional_losses_1019360

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
Қ
�
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021220
lambda_6_input;
-batch_normalization_6_readvariableop_resource:=
/batch_normalization_6_readvariableop_1_resource:L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_18_conv2d_readvariableop_resource: 7
)conv2d_18_biasadd_readvariableop_resource: C
(conv2d_19_conv2d_readvariableop_resource: �8
)conv2d_19_biasadd_readvariableop_resource:	�D
(conv2d_20_conv2d_readvariableop_resource:��8
)conv2d_20_biasadd_readvariableop_resource:	�<
'dense_18_matmul_readvariableop_resource:���7
(dense_18_biasadd_readvariableop_resource:	�;
'dense_19_matmul_readvariableop_resource:
��7
(dense_19_biasadd_readvariableop_resource:	�
identity��$batch_normalization_6/AssignNewValue�&batch_normalization_6/AssignNewValue_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�1dense_19/kernel/Regularizer/Square/ReadVariableOp�
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
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3lambda_6/strided_slice:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_6/FusedBatchNormV3�
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue�
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_18/Conv2D�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_18/BiasAdd~
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_18/Relu�
max_pooling2d_18/MaxPoolMaxPoolconv2d_18/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_18/MaxPool�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_19/Conv2D�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_19/Relu�
max_pooling2d_19/MaxPoolMaxPoolconv2d_19/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_19/MaxPool�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_20/Conv2D/ReadVariableOp�
conv2d_20/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_20/Conv2D�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_20/BiasAdd
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_20/Relu�
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPooly
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_18/dropout/Const�
dropout_18/dropout/MulMul!max_pooling2d_20/MaxPool:output:0!dropout_18/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2
dropout_18/dropout/Mul�
dropout_18/dropout/ShapeShape!max_pooling2d_20/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shape�
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype021
/dropout_18/dropout/random_uniform/RandomUniform�
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_18/dropout/GreaterEqual/y�
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2!
dropout_18/dropout/GreaterEqual�
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout_18/dropout/Cast�
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout_18/dropout/Mul_1s
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_6/Const�
flatten_6/ReshapeReshapedropout_18/dropout/Mul_1:z:0flatten_6/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_6/Reshape�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
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
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_19/dropout/Const�
dropout_19/dropout/MulMuldense_18/Relu:activations:0!dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_19/dropout/Mul
dropout_19/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shape�
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_19/dropout/random_uniform/RandomUniform�
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_19/dropout/GreaterEqual/y�
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_19/dropout/GreaterEqual�
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_19/dropout/Cast�
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_19/dropout/Mul_1�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_19/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
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
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_20/dropout/Const�
dropout_20/dropout/MulMuldense_19/Relu:activations:0!dropout_20/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_20/dropout/Mul
dropout_20/dropout/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shape�
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_20/dropout/random_uniform/RandomUniform�
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_20/dropout/GreaterEqual/y�
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_20/dropout/GreaterEqual�
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_20/dropout/Cast�
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_20/dropout/Mul_1�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_18/kernel/Regularizer/Square�
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_18/kernel/Regularizer/Const�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/Sum�
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_18/kernel/Regularizer/mul/x�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_18/kernel/Regularizer/mul�
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
dense_19/kernel/Regularizer/mul�
IdentityIdentitydropout_20/dropout/Mul_1:z:0%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^dense_19/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2B
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
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021452

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
�
E__inference_dense_18_layer_call_and_return_conditional_losses_1021655

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_18/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
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
:���*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOp�
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
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
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_1019432

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
:���������		�2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
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
:���������		�2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������		�:X T
0
_output_shapes
:���������		�
 
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
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__
	�call"�	
_tf_keras_model�{"name": "CNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�h
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
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�d
_tf_keras_sequential�d{"name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_6_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_6_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_6_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem�m�&m�'m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�v�v�&v�'v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�"
	optimizer
 "
trackable_list_wrapper
�
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
14
15"
trackable_list_wrapper
�
&0
'1
*2
+3
,4
-5
.6
/7
08
19
210
311
12
13"
trackable_list_wrapper
�
regularization_losses
4non_trainable_variables
	variables
5layer_regularization_losses
6metrics

7layers
8layer_metrics
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "lambda_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_6", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
�

=axis
	&gamma
'beta
(moving_mean
)moving_variance
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

*kernel
+bias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
�


,kernel
-bias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
�
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
�


.kernel
/bias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
�
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
�
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
�
^regularization_losses
_	variables
`trainable_variables
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
�	

0kernel
1bias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
�
fregularization_losses
g	variables
htrainable_variables
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
�	

2kernel
3bias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
8
�0
�1
�2"
trackable_list_wrapper
�
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313"
trackable_list_wrapper
v
&0
'1
*2
+3
,4
-5
.6
/7
08
19
210
311"
trackable_list_wrapper
�
regularization_losses
rnon_trainable_variables
	variables
slayer_regularization_losses
tmetrics

ulayers
vlayer_metrics
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_20/kernel
:2dense_20/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
wnon_trainable_variables
regularization_losses
xlayer_regularization_losses
	variables
ymetrics

zlayers
{layer_metrics
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
*:( 2conv2d_18/kernel
: 2conv2d_18/bias
+:) �2conv2d_19/kernel
:�2conv2d_19/bias
,:*��2conv2d_20/kernel
:�2conv2d_20/bias
$:"���2dense_18/kernel
:�2dense_18/bias
#:!
��2dense_19/kernel
:�2dense_19/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
0
1"
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
~non_trainable_variables
9regularization_losses
layer_regularization_losses
:	variables
�metrics
�layers
�layer_metrics
;trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
�non_trainable_variables
>regularization_losses
 �layer_regularization_losses
?	variables
�metrics
�layers
�layer_metrics
@trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
�
�non_trainable_variables
Bregularization_losses
 �layer_regularization_losses
C	variables
�metrics
�layers
�layer_metrics
Dtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Fregularization_losses
 �layer_regularization_losses
G	variables
�metrics
�layers
�layer_metrics
Htrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
�
�non_trainable_variables
Jregularization_losses
 �layer_regularization_losses
K	variables
�metrics
�layers
�layer_metrics
Ltrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Nregularization_losses
 �layer_regularization_losses
O	variables
�metrics
�layers
�layer_metrics
Ptrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
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
�non_trainable_variables
Rregularization_losses
 �layer_regularization_losses
S	variables
�metrics
�layers
�layer_metrics
Ttrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Vregularization_losses
 �layer_regularization_losses
W	variables
�metrics
�layers
�layer_metrics
Xtrainable_variables
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
Zregularization_losses
 �layer_regularization_losses
[	variables
�metrics
�layers
�layer_metrics
\trainable_variables
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
^regularization_losses
 �layer_regularization_losses
_	variables
�metrics
�layers
�layer_metrics
`trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
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
�non_trainable_variables
bregularization_losses
 �layer_regularization_losses
c	variables
�metrics
�layers
�layer_metrics
dtrainable_variables
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
fregularization_losses
 �layer_regularization_losses
g	variables
�metrics
�layers
�layer_metrics
htrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
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
�non_trainable_variables
jregularization_losses
 �layer_regularization_losses
k	variables
�metrics
�layers
�layer_metrics
ltrainable_variables
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
nregularization_losses
 �layer_regularization_losses
o	variables
�metrics
�layers
�layer_metrics
ptrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
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
13"
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
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
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
.
(0
)1"
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
(
�0"
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
(
�0"
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
(
�0"
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
.:,2"Adam/batch_normalization_6/gamma/m
-:+2!Adam/batch_normalization_6/beta/m
/:- 2Adam/conv2d_18/kernel/m
!: 2Adam/conv2d_18/bias/m
0:. �2Adam/conv2d_19/kernel/m
": �2Adam/conv2d_19/bias/m
1:/��2Adam/conv2d_20/kernel/m
": �2Adam/conv2d_20/bias/m
):'���2Adam/dense_18/kernel/m
!:�2Adam/dense_18/bias/m
(:&
��2Adam/dense_19/kernel/m
!:�2Adam/dense_19/bias/m
':%	�2Adam/dense_20/kernel/v
 :2Adam/dense_20/bias/v
.:,2"Adam/batch_normalization_6/gamma/v
-:+2!Adam/batch_normalization_6/beta/v
/:- 2Adam/conv2d_18/kernel/v
!: 2Adam/conv2d_18/bias/v
0:. �2Adam/conv2d_19/kernel/v
": �2Adam/conv2d_19/bias/v
1:/��2Adam/conv2d_20/kernel/v
": �2Adam/conv2d_20/bias/v
):'���2Adam/dense_18/kernel/v
!:�2Adam/dense_18/bias/v
(:&
��2Adam/dense_19/kernel/v
!:�2Adam/dense_19/bias/v
�2�
@__inference_CNN_layer_call_and_return_conditional_losses_1020368
@__inference_CNN_layer_call_and_return_conditional_losses_1020479
@__inference_CNN_layer_call_and_return_conditional_losses_1020569
@__inference_CNN_layer_call_and_return_conditional_losses_1020680�
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
"__inference__wrapped_model_1018949�
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
%__inference_CNN_layer_call_fn_1020717
%__inference_CNN_layer_call_fn_1020754
%__inference_CNN_layer_call_fn_1020791
%__inference_CNN_layer_call_fn_1020828�
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
__inference_call_939328
__inference_call_939400
__inference_call_939472�
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_1020929
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021033
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021116
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021220�
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
.__inference_sequential_6_layer_call_fn_1021253
.__inference_sequential_6_layer_call_fn_1021286
.__inference_sequential_6_layer_call_fn_1021319
.__inference_sequential_6_layer_call_fn_1021352�
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
E__inference_dense_20_layer_call_and_return_conditional_losses_1021363�
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
*__inference_dense_20_layer_call_fn_1021372�
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
%__inference_signature_wrapper_1020278input_1"�
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
E__inference_lambda_6_layer_call_and_return_conditional_losses_1021380
E__inference_lambda_6_layer_call_and_return_conditional_losses_1021388�
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
*__inference_lambda_6_layer_call_fn_1021393
*__inference_lambda_6_layer_call_fn_1021398�
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021416
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021434
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021452
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021470�
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
7__inference_batch_normalization_6_layer_call_fn_1021483
7__inference_batch_normalization_6_layer_call_fn_1021496
7__inference_batch_normalization_6_layer_call_fn_1021509
7__inference_batch_normalization_6_layer_call_fn_1021522�
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
F__inference_conv2d_18_layer_call_and_return_conditional_losses_1021545�
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
+__inference_conv2d_18_layer_call_fn_1021554�
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
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_1019081�
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
2__inference_max_pooling2d_18_layer_call_fn_1019087�
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
F__inference_conv2d_19_layer_call_and_return_conditional_losses_1021565�
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
+__inference_conv2d_19_layer_call_fn_1021574�
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
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_1019093�
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
2__inference_max_pooling2d_19_layer_call_fn_1019099�
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
F__inference_conv2d_20_layer_call_and_return_conditional_losses_1021585�
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
+__inference_conv2d_20_layer_call_fn_1021594�
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
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1019105�
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
2__inference_max_pooling2d_20_layer_call_fn_1019111�
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
G__inference_dropout_18_layer_call_and_return_conditional_losses_1021599
G__inference_dropout_18_layer_call_and_return_conditional_losses_1021611�
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
,__inference_dropout_18_layer_call_fn_1021616
,__inference_dropout_18_layer_call_fn_1021621�
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
F__inference_flatten_6_layer_call_and_return_conditional_losses_1021627�
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
+__inference_flatten_6_layer_call_fn_1021632�
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
E__inference_dense_18_layer_call_and_return_conditional_losses_1021655�
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
*__inference_dense_18_layer_call_fn_1021664�
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
G__inference_dropout_19_layer_call_and_return_conditional_losses_1021669
G__inference_dropout_19_layer_call_and_return_conditional_losses_1021681�
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
,__inference_dropout_19_layer_call_fn_1021686
,__inference_dropout_19_layer_call_fn_1021691�
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
E__inference_dense_19_layer_call_and_return_conditional_losses_1021714�
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
*__inference_dense_19_layer_call_fn_1021723�
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
G__inference_dropout_20_layer_call_and_return_conditional_losses_1021728
G__inference_dropout_20_layer_call_and_return_conditional_losses_1021740�
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
,__inference_dropout_20_layer_call_fn_1021745
,__inference_dropout_20_layer_call_fn_1021750�
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
__inference_loss_fn_0_1021761�
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
__inference_loss_fn_1_1021772�
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
__inference_loss_fn_2_1021783�
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
@__inference_CNN_layer_call_and_return_conditional_losses_1020368v&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p 
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_1020479v&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_1020569w&'()*+,-./0123<�9
2�/
)�&
input_1���������KK
p 
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_1020680w&'()*+,-./0123<�9
2�/
)�&
input_1���������KK
p
� "%�"
�
0���������
� �
%__inference_CNN_layer_call_fn_1020717j&'()*+,-./0123<�9
2�/
)�&
input_1���������KK
p 
� "�����������
%__inference_CNN_layer_call_fn_1020754i&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p 
� "�����������
%__inference_CNN_layer_call_fn_1020791i&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p
� "�����������
%__inference_CNN_layer_call_fn_1020828j&'()*+,-./0123<�9
2�/
)�&
input_1���������KK
p
� "�����������
"__inference__wrapped_model_1018949�&'()*+,-./01238�5
.�+
)�&
input_1���������KK
� "3�0
.
output_1"�
output_1����������
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021416�&'()M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021434�&'()M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021452r&'();�8
1�.
(�%
inputs���������KK
p 
� "-�*
#� 
0���������KK
� �
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1021470r&'();�8
1�.
(�%
inputs���������KK
p
� "-�*
#� 
0���������KK
� �
7__inference_batch_normalization_6_layer_call_fn_1021483�&'()M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
7__inference_batch_normalization_6_layer_call_fn_1021496�&'()M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
7__inference_batch_normalization_6_layer_call_fn_1021509e&'();�8
1�.
(�%
inputs���������KK
p 
� " ����������KK�
7__inference_batch_normalization_6_layer_call_fn_1021522e&'();�8
1�.
(�%
inputs���������KK
p
� " ����������KKt
__inference_call_939328Y&'()*+,-./01233�0
)�&
 �
inputs�KK
p
� "�	�t
__inference_call_939400Y&'()*+,-./01233�0
)�&
 �
inputs�KK
p 
� "�	��
__inference_call_939472i&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p 
� "�����������
F__inference_conv2d_18_layer_call_and_return_conditional_losses_1021545l*+7�4
-�*
(�%
inputs���������KK
� "-�*
#� 
0���������KK 
� �
+__inference_conv2d_18_layer_call_fn_1021554_*+7�4
-�*
(�%
inputs���������KK
� " ����������KK �
F__inference_conv2d_19_layer_call_and_return_conditional_losses_1021565m,-7�4
-�*
(�%
inputs���������%% 
� ".�+
$�!
0���������%%�
� �
+__inference_conv2d_19_layer_call_fn_1021574`,-7�4
-�*
(�%
inputs���������%% 
� "!����������%%��
F__inference_conv2d_20_layer_call_and_return_conditional_losses_1021585n./8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_conv2d_20_layer_call_fn_1021594a./8�5
.�+
)�&
inputs����������
� "!������������
E__inference_dense_18_layer_call_and_return_conditional_losses_1021655_011�.
'�$
"�
inputs�����������
� "&�#
�
0����������
� �
*__inference_dense_18_layer_call_fn_1021664R011�.
'�$
"�
inputs�����������
� "������������
E__inference_dense_19_layer_call_and_return_conditional_losses_1021714^230�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_19_layer_call_fn_1021723Q230�-
&�#
!�
inputs����������
� "������������
E__inference_dense_20_layer_call_and_return_conditional_losses_1021363]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_20_layer_call_fn_1021372P0�-
&�#
!�
inputs����������
� "�����������
G__inference_dropout_18_layer_call_and_return_conditional_losses_1021599n<�9
2�/
)�&
inputs���������		�
p 
� ".�+
$�!
0���������		�
� �
G__inference_dropout_18_layer_call_and_return_conditional_losses_1021611n<�9
2�/
)�&
inputs���������		�
p
� ".�+
$�!
0���������		�
� �
,__inference_dropout_18_layer_call_fn_1021616a<�9
2�/
)�&
inputs���������		�
p 
� "!����������		��
,__inference_dropout_18_layer_call_fn_1021621a<�9
2�/
)�&
inputs���������		�
p
� "!����������		��
G__inference_dropout_19_layer_call_and_return_conditional_losses_1021669^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_19_layer_call_and_return_conditional_losses_1021681^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_19_layer_call_fn_1021686Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_19_layer_call_fn_1021691Q4�1
*�'
!�
inputs����������
p
� "������������
G__inference_dropout_20_layer_call_and_return_conditional_losses_1021728^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_20_layer_call_and_return_conditional_losses_1021740^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_20_layer_call_fn_1021745Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_20_layer_call_fn_1021750Q4�1
*�'
!�
inputs����������
p
� "������������
F__inference_flatten_6_layer_call_and_return_conditional_losses_1021627c8�5
.�+
)�&
inputs���������		�
� "'�$
�
0�����������
� �
+__inference_flatten_6_layer_call_fn_1021632V8�5
.�+
)�&
inputs���������		�
� "�������������
E__inference_lambda_6_layer_call_and_return_conditional_losses_1021380p?�<
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
E__inference_lambda_6_layer_call_and_return_conditional_losses_1021388p?�<
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
*__inference_lambda_6_layer_call_fn_1021393c?�<
5�2
(�%
inputs���������KK

 
p 
� " ����������KK�
*__inference_lambda_6_layer_call_fn_1021398c?�<
5�2
(�%
inputs���������KK

 
p
� " ����������KK<
__inference_loss_fn_0_1021761*�

� 
� "� <
__inference_loss_fn_1_10217720�

� 
� "� <
__inference_loss_fn_2_10217832�

� 
� "� �
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_1019081�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_18_layer_call_fn_1019087�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_1019093�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_19_layer_call_fn_1019099�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1019105�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_20_layer_call_fn_1019111�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_6_layer_call_and_return_conditional_losses_1020929y&'()*+,-./0123?�<
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021033y&'()*+,-./0123?�<
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021116�&'()*+,-./0123G�D
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_1021220�&'()*+,-./0123G�D
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
.__inference_sequential_6_layer_call_fn_1021253t&'()*+,-./0123G�D
=�:
0�-
lambda_6_input���������KK
p 

 
� "������������
.__inference_sequential_6_layer_call_fn_1021286l&'()*+,-./0123?�<
5�2
(�%
inputs���������KK
p 

 
� "������������
.__inference_sequential_6_layer_call_fn_1021319l&'()*+,-./0123?�<
5�2
(�%
inputs���������KK
p

 
� "������������
.__inference_sequential_6_layer_call_fn_1021352t&'()*+,-./0123G�D
=�:
0�-
lambda_6_input���������KK
p

 
� "������������
%__inference_signature_wrapper_1020278�&'()*+,-./0123C�@
� 
9�6
4
input_1)�&
input_1���������KK"3�0
.
output_1"�
output_1���������