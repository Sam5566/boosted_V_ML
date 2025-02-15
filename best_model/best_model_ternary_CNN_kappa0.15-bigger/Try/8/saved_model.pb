��
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
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_26/kernel
t
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes
:	�*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:*
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
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma
�
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta
�
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean
�
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance
�
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
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
}
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���* 
shared_namedense_24/kernel
v
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*!
_output_shapes
:���*
dtype0
s
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_24/bias
l
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes	
:�*
dtype0
|
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_25/kernel
u
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel* 
_output_shapes
:
��*
dtype0
s
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_25/bias
l
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
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
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_26/kernel/m
�
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_8/gamma/m
�
6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/m*
_output_shapes
:*
dtype0
�
!Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_8/beta/m
�
5Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/m*
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
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*'
shared_nameAdam/dense_24/kernel/m
�
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*!
_output_shapes
:���*
dtype0
�
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_24/bias/m
z
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_25/kernel/m
�
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_25/bias/m
z
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_26/kernel/v
�
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_8/gamma/v
�
6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/v*
_output_shapes
:*
dtype0
�
!Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_8/beta/v
�
5Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/v*
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
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*'
shared_nameAdam/dense_24/kernel/v
�
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*!
_output_shapes
:���*
dtype0
�
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_24/bias/v
z
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_25/kernel/v
�
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_25/bias/v
z
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
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
	variables
regularization_losses
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
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem�m�&m�'m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�v�v�&v�'v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�
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
 
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
	variables
regularization_losses
trainable_variables
4metrics
5layer_regularization_losses
6layer_metrics

7layers
8non_trainable_variables
 
R
9	variables
:regularization_losses
;trainable_variables
<	keras_api
�
=axis
	&gamma
'beta
(moving_mean
)moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
h

*kernel
+bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
R
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
h

,kernel
-bias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
R
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
h

.kernel
/bias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
R
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
R
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
R
^	variables
_regularization_losses
`trainable_variables
a	keras_api
h

0kernel
1bias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
R
f	variables
gregularization_losses
htrainable_variables
i	keras_api
h

2kernel
3bias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
R
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
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
 
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
	variables
regularization_losses
trainable_variables
rmetrics
slayer_regularization_losses
tlayer_metrics

ulayers
vnon_trainable_variables
NL
VARIABLE_VALUEdense_26/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_26/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
regularization_losses
wlayer_regularization_losses
xmetrics
trainable_variables
ylayer_metrics

zlayers
{non_trainable_variables
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
VARIABLE_VALUEbatch_normalization_8/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_8/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_8/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_8/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_24/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_24/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_25/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_25/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_26/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_26/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_24/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_24/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_25/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_25/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE

|0
}1
 
 

0
1

(0
)1
 
 
 
�
9	variables
:regularization_losses
~layer_regularization_losses
metrics
;trainable_variables
�layer_metrics
�layers
�non_trainable_variables
 

&0
'1
(2
)3
 

&0
'1
�
>	variables
?regularization_losses
 �layer_regularization_losses
�metrics
@trainable_variables
�layer_metrics
�layers
�non_trainable_variables

*0
+1
 

*0
+1
�
B	variables
Cregularization_losses
 �layer_regularization_losses
�metrics
Dtrainable_variables
�layer_metrics
�layers
�non_trainable_variables
 
 
 
�
F	variables
Gregularization_losses
 �layer_regularization_losses
�metrics
Htrainable_variables
�layer_metrics
�layers
�non_trainable_variables

,0
-1
 

,0
-1
�
J	variables
Kregularization_losses
 �layer_regularization_losses
�metrics
Ltrainable_variables
�layer_metrics
�layers
�non_trainable_variables
 
 
 
�
N	variables
Oregularization_losses
 �layer_regularization_losses
�metrics
Ptrainable_variables
�layer_metrics
�layers
�non_trainable_variables

.0
/1
 

.0
/1
�
R	variables
Sregularization_losses
 �layer_regularization_losses
�metrics
Ttrainable_variables
�layer_metrics
�layers
�non_trainable_variables
 
 
 
�
V	variables
Wregularization_losses
 �layer_regularization_losses
�metrics
Xtrainable_variables
�layer_metrics
�layers
�non_trainable_variables
 
 
 
�
Z	variables
[regularization_losses
 �layer_regularization_losses
�metrics
\trainable_variables
�layer_metrics
�layers
�non_trainable_variables
 
 
 
�
^	variables
_regularization_losses
 �layer_regularization_losses
�metrics
`trainable_variables
�layer_metrics
�layers
�non_trainable_variables

00
11
 

00
11
�
b	variables
cregularization_losses
 �layer_regularization_losses
�metrics
dtrainable_variables
�layer_metrics
�layers
�non_trainable_variables
 
 
 
�
f	variables
gregularization_losses
 �layer_regularization_losses
�metrics
htrainable_variables
�layer_metrics
�layers
�non_trainable_variables

20
31
 

20
31
�
j	variables
kregularization_losses
 �layer_regularization_losses
�metrics
ltrainable_variables
�layer_metrics
�layers
�non_trainable_variables
 
 
 
�
n	variables
oregularization_losses
 �layer_regularization_losses
�metrics
ptrainable_variables
�layer_metrics
�layers
�non_trainable_variables
 
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

(0
)1
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
VARIABLE_VALUEAdam/dense_26/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_26/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_8/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_24/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_24/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_25/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_25/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_26/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_26/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_24/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_24/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_25/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_25/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_26/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_26/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_8/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_24/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_24/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_25/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_25/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_26/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_26/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_24/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_24/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_25/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_25/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������KK*
dtype0*$
shape:���������KK
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_7365426
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_8/beta/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp+Adam/conv2d_25/kernel/m/Read/ReadVariableOp)Adam/conv2d_25/bias/m/Read/ReadVariableOp+Adam/conv2d_26/kernel/m/Read/ReadVariableOp)Adam/conv2d_26/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_8/beta/v/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp+Adam/conv2d_25/kernel/v/Read/ReadVariableOp)Adam/conv2d_25/bias/v/Read/ReadVariableOp+Adam/conv2d_26/kernel/v/Read/ReadVariableOp)Adam/conv2d_26/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOpConst*B
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
 __inference__traced_save_7367113
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_26/kerneldense_26/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biastotalcounttotal_1count_1Adam/dense_26/kernel/mAdam/dense_26/bias/m"Adam/batch_normalization_8/gamma/m!Adam/batch_normalization_8/beta/mAdam/conv2d_24/kernel/mAdam/conv2d_24/bias/mAdam/conv2d_25/kernel/mAdam/conv2d_25/bias/mAdam/conv2d_26/kernel/mAdam/conv2d_26/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/mAdam/dense_26/kernel/vAdam/dense_26/bias/v"Adam/batch_normalization_8/gamma/v!Adam/batch_normalization_8/beta/vAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/vAdam/conv2d_25/kernel/vAdam/conv2d_25/bias/vAdam/conv2d_26/kernel/vAdam/conv2d_26/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/v*A
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
#__inference__traced_restore_7367282��
�2
�
@__inference_CNN_layer_call_and_return_conditional_losses_7365039

inputs"
sequential_8_7364974:"
sequential_8_7364976:"
sequential_8_7364978:"
sequential_8_7364980:.
sequential_8_7364982: "
sequential_8_7364984: /
sequential_8_7364986: �#
sequential_8_7364988:	�0
sequential_8_7364990:��#
sequential_8_7364992:	�)
sequential_8_7364994:���#
sequential_8_7364996:	�(
sequential_8_7364998:
��#
sequential_8_7365000:	�#
dense_26_7365015:	�
dense_26_7365017:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp� dense_26/StatefulPartitionedCall�$sequential_8/StatefulPartitionedCall�
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinputssequential_8_7364974sequential_8_7364976sequential_8_7364978sequential_8_7364980sequential_8_7364982sequential_8_7364984sequential_8_7364986sequential_8_7364988sequential_8_7364990sequential_8_7364992sequential_8_7364994sequential_8_7364996sequential_8_7364998sequential_8_7365000*
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_73644572&
$sequential_8/StatefulPartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0dense_26_7365015dense_26_7365017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_73650142"
 dense_26/StatefulPartitionedCall�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_8_7364982*&
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_8_7364994*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_8_7364998* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentity)dense_26/StatefulPartitionedCall:output:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
F__inference_conv2d_26_layer_call_and_return_conditional_losses_7366733

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
�s
�
__inference_call_7256974

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	�Q
5sequential_8_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	�I
4sequential_8_dense_24_matmul_readvariableop_resource:���D
5sequential_8_dense_24_biasadd_readvariableop_resource:	�H
4sequential_8_dense_25_matmul_readvariableop_resource:
��D
5sequential_8_dense_25_biasadd_readvariableop_resource:	�:
'dense_26_matmul_readvariableop_resource:	�6
(dense_26_biasadd_readvariableop_resource:
identity��dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_8/batch_normalization_8/ReadVariableOp�3sequential_8/batch_normalization_8/ReadVariableOp_1�-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�,sequential_8/conv2d_24/Conv2D/ReadVariableOp�-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�,sequential_8/conv2d_25/Conv2D/ReadVariableOp�-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�,sequential_8/conv2d_26/Conv2D/ReadVariableOp�,sequential_8/dense_24/BiasAdd/ReadVariableOp�+sequential_8/dense_24/MatMul/ReadVariableOp�,sequential_8/dense_25/BiasAdd/ReadVariableOp�+sequential_8/dense_25/MatMul/ReadVariableOp�
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_8/lambda_8/strided_slice/stack�
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_8/lambda_8/strided_slice/stack_1�
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_8/lambda_8/strided_slice/stack_2�
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_8/lambda_8/strided_slice�
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp�
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3�
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_24/Conv2D/ReadVariableOp�
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_8/conv2d_24/Conv2D�
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_8/conv2d_24/BiasAdd�
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_8/conv2d_24/Relu�
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_24/MaxPool�
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_8/conv2d_25/Conv2D/ReadVariableOp�
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_8/conv2d_25/Conv2D�
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_8/conv2d_25/BiasAdd�
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_8/conv2d_25/Relu�
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_25/MaxPool�
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_8/conv2d_26/Conv2D/ReadVariableOp�
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_8/conv2d_26/Conv2D�
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_8/conv2d_26/BiasAdd�
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_8/conv2d_26/Relu�
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_26/MaxPool�
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*(
_output_shapes
:�		�2"
 sequential_8/dropout_24/Identity�
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_8/flatten_8/Const�
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*!
_output_shapes
:���2 
sequential_8/flatten_8/Reshape�
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_8/dense_24/MatMul/ReadVariableOp�
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_24/MatMul�
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_24/BiasAdd/ReadVariableOp�
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_24/BiasAdd�
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_24/Relu�
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_8/dropout_25/Identity�
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_8/dense_25/MatMul/ReadVariableOp�
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_25/MatMul�
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_25/BiasAdd/ReadVariableOp�
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_25/BiasAdd�
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_25/Relu�
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_8/dropout_26/Identity�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_26/BiasAddt
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_26/Softmax�
IdentityIdentitydense_26/Softmax:softmax:0 ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:�KK: : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
f
G__inference_dropout_25_layer_call_and_return_conditional_losses_7366829

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
,__inference_dropout_25_layer_call_fn_7366834

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
G__inference_dropout_25_layer_call_and_return_conditional_losses_73644062
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
.__inference_sequential_8_layer_call_fn_7366500
lambda_8_input
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
StatefulPartitionedCallStatefulPartitionedCalllambda_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_73647752
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
_user_specified_namelambda_8_input
�
N
2__inference_max_pooling2d_24_layer_call_fn_7364235

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
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_73642292
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
�s
�
__inference_call_7257046

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	�Q
5sequential_8_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	�I
4sequential_8_dense_24_matmul_readvariableop_resource:���D
5sequential_8_dense_24_biasadd_readvariableop_resource:	�H
4sequential_8_dense_25_matmul_readvariableop_resource:
��D
5sequential_8_dense_25_biasadd_readvariableop_resource:	�:
'dense_26_matmul_readvariableop_resource:	�6
(dense_26_biasadd_readvariableop_resource:
identity��dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_8/batch_normalization_8/ReadVariableOp�3sequential_8/batch_normalization_8/ReadVariableOp_1�-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�,sequential_8/conv2d_24/Conv2D/ReadVariableOp�-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�,sequential_8/conv2d_25/Conv2D/ReadVariableOp�-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�,sequential_8/conv2d_26/Conv2D/ReadVariableOp�,sequential_8/dense_24/BiasAdd/ReadVariableOp�+sequential_8/dense_24/MatMul/ReadVariableOp�,sequential_8/dense_25/BiasAdd/ReadVariableOp�+sequential_8/dense_25/MatMul/ReadVariableOp�
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_8/lambda_8/strided_slice/stack�
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_8/lambda_8/strided_slice/stack_1�
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_8/lambda_8/strided_slice/stack_2�
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_8/lambda_8/strided_slice�
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp�
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3�
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_24/Conv2D/ReadVariableOp�
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_8/conv2d_24/Conv2D�
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_8/conv2d_24/BiasAdd�
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_8/conv2d_24/Relu�
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_24/MaxPool�
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_8/conv2d_25/Conv2D/ReadVariableOp�
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_8/conv2d_25/Conv2D�
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_8/conv2d_25/BiasAdd�
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_8/conv2d_25/Relu�
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_25/MaxPool�
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_8/conv2d_26/Conv2D/ReadVariableOp�
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_8/conv2d_26/Conv2D�
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_8/conv2d_26/BiasAdd�
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_8/conv2d_26/Relu�
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_26/MaxPool�
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*(
_output_shapes
:�		�2"
 sequential_8/dropout_24/Identity�
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_8/flatten_8/Const�
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*!
_output_shapes
:���2 
sequential_8/flatten_8/Reshape�
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_8/dense_24/MatMul/ReadVariableOp�
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_24/MatMul�
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_24/BiasAdd/ReadVariableOp�
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_24/BiasAdd�
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_24/Relu�
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_8/dropout_25/Identity�
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_8/dense_25/MatMul/ReadVariableOp�
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_25/MatMul�
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_25/BiasAdd/ReadVariableOp�
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_25/BiasAdd�
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_8/dense_25/Relu�
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_8/dropout_26/Identity�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_26/BiasAddt
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_26/Softmax�
IdentityIdentitydense_26/Softmax:softmax:0 ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:�KK: : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
�
%__inference_CNN_layer_call_fn_7365976
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

unknown_13:	�

unknown_14:
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
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_73651792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�
�
.__inference_sequential_8_layer_call_fn_7366434

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
I__inference_sequential_8_layer_call_and_return_conditional_losses_73644572
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
�
7__inference_batch_normalization_8_layer_call_fn_7366644

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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_73641632
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
f
G__inference_dropout_24_layer_call_and_return_conditional_losses_7364580

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
�
e
G__inference_dropout_26_layer_call_and_return_conditional_losses_7364436

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
G__inference_dropout_26_layer_call_and_return_conditional_losses_7364508

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
�
@__inference_CNN_layer_call_and_return_conditional_losses_7365717
input_1H
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	�Q
5sequential_8_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	�I
4sequential_8_dense_24_matmul_readvariableop_resource:���D
5sequential_8_dense_24_biasadd_readvariableop_resource:	�H
4sequential_8_dense_25_matmul_readvariableop_resource:
��D
5sequential_8_dense_25_biasadd_readvariableop_resource:	�:
'dense_26_matmul_readvariableop_resource:	�6
(dense_26_biasadd_readvariableop_resource:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_8/batch_normalization_8/ReadVariableOp�3sequential_8/batch_normalization_8/ReadVariableOp_1�-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�,sequential_8/conv2d_24/Conv2D/ReadVariableOp�-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�,sequential_8/conv2d_25/Conv2D/ReadVariableOp�-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�,sequential_8/conv2d_26/Conv2D/ReadVariableOp�,sequential_8/dense_24/BiasAdd/ReadVariableOp�+sequential_8/dense_24/MatMul/ReadVariableOp�,sequential_8/dense_25/BiasAdd/ReadVariableOp�+sequential_8/dense_25/MatMul/ReadVariableOp�
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_8/lambda_8/strided_slice/stack�
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_8/lambda_8/strided_slice/stack_1�
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_8/lambda_8/strided_slice/stack_2�
#sequential_8/lambda_8/strided_sliceStridedSliceinput_12sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_8/lambda_8/strided_slice�
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp�
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3�
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_24/Conv2D/ReadVariableOp�
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_8/conv2d_24/Conv2D�
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_8/conv2d_24/BiasAdd�
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_8/conv2d_24/Relu�
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_24/MaxPool�
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_8/conv2d_25/Conv2D/ReadVariableOp�
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_8/conv2d_25/Conv2D�
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_8/conv2d_25/BiasAdd�
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_8/conv2d_25/Relu�
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_25/MaxPool�
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_8/conv2d_26/Conv2D/ReadVariableOp�
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_8/conv2d_26/Conv2D�
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_8/conv2d_26/BiasAdd�
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_8/conv2d_26/Relu�
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_26/MaxPool�
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_8/dropout_24/Identity�
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_8/flatten_8/Const�
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_8/flatten_8/Reshape�
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_8/dense_24/MatMul/ReadVariableOp�
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/MatMul�
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_24/BiasAdd/ReadVariableOp�
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/BiasAdd�
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/Relu�
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_8/dropout_25/Identity�
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_8/dense_25/MatMul/ReadVariableOp�
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/MatMul�
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_25/BiasAdd/ReadVariableOp�
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/BiasAdd�
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/Relu�
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_8/dropout_26/Identity�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/BiasAdd|
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_26/Softmax�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentitydense_26/Softmax:softmax:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
e
,__inference_dropout_24_layer_call_fn_7366769

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
G__inference_dropout_24_layer_call_and_return_conditional_losses_73645802
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
�
F
*__inference_lambda_8_layer_call_fn_7366546

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
E__inference_lambda_8_layer_call_and_return_conditional_losses_73646732
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
�
__inference_loss_fn_0_7366909U
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
�
f
G__inference_dropout_26_layer_call_and_return_conditional_losses_7366888

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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366582

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
�
%__inference_signature_wrapper_7365426
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

unknown_13:	�

unknown_14:
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
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__wrapped_model_73640972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�a
�	
I__inference_sequential_8_layer_call_and_return_conditional_losses_7364775

inputs+
batch_normalization_8_7364715:+
batch_normalization_8_7364717:+
batch_normalization_8_7364719:+
batch_normalization_8_7364721:+
conv2d_24_7364724: 
conv2d_24_7364726: ,
conv2d_25_7364730: � 
conv2d_25_7364732:	�-
conv2d_26_7364736:�� 
conv2d_26_7364738:	�%
dense_24_7364744:���
dense_24_7364746:	�$
dense_25_7364750:
��
dense_25_7364752:	�
identity��-batch_normalization_8/StatefulPartitionedCall�!conv2d_24/StatefulPartitionedCall�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�!conv2d_25/StatefulPartitionedCall�!conv2d_26/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�1dense_24/kernel/Regularizer/Square/ReadVariableOp� dense_25/StatefulPartitionedCall�1dense_25/kernel/Regularizer/Square/ReadVariableOp�"dropout_24/StatefulPartitionedCall�"dropout_25/StatefulPartitionedCall�"dropout_26/StatefulPartitionedCall�
lambda_8/PartitionedCallPartitionedCallinputs*
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
E__inference_lambda_8_layer_call_and_return_conditional_losses_73646732
lambda_8/PartitionedCall�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0batch_normalization_8_7364715batch_normalization_8_7364717batch_normalization_8_7364719batch_normalization_8_7364721*
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_73646462/
-batch_normalization_8/StatefulPartitionedCall�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_24_7364724conv2d_24_7364726*
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
F__inference_conv2d_24_layer_call_and_return_conditional_losses_73643202#
!conv2d_24/StatefulPartitionedCall�
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_73642292"
 max_pooling2d_24/PartitionedCall�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_7364730conv2d_25_7364732*
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
F__inference_conv2d_25_layer_call_and_return_conditional_losses_73643382#
!conv2d_25/StatefulPartitionedCall�
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_73642412"
 max_pooling2d_25/PartitionedCall�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_7364736conv2d_26_7364738*
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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_73643562#
!conv2d_26/StatefulPartitionedCall�
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_73642532"
 max_pooling2d_26/PartitionedCall�
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
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
G__inference_dropout_24_layer_call_and_return_conditional_losses_73645802$
"dropout_24/StatefulPartitionedCall�
flatten_8/PartitionedCallPartitionedCall+dropout_24/StatefulPartitionedCall:output:0*
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
F__inference_flatten_8_layer_call_and_return_conditional_losses_73643762
flatten_8/PartitionedCall�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_24_7364744dense_24_7364746*
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
E__inference_dense_24_layer_call_and_return_conditional_losses_73643952"
 dense_24/StatefulPartitionedCall�
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
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
G__inference_dropout_25_layer_call_and_return_conditional_losses_73645412$
"dropout_25/StatefulPartitionedCall�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_25_7364750dense_25_7364752*
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
E__inference_dense_25_layer_call_and_return_conditional_losses_73644252"
 dense_25/StatefulPartitionedCall�
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
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
G__inference_dropout_26_layer_call_and_return_conditional_losses_73645082$
"dropout_26/StatefulPartitionedCall�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_7364724*&
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_24_7364744*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_7364750* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentity+dropout_26/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/Square/ReadVariableOp#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
G
+__inference_flatten_8_layer_call_fn_7366780

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
F__inference_flatten_8_layer_call_and_return_conditional_losses_73643762
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
�
�
E__inference_dense_24_layer_call_and_return_conditional_losses_7366803

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_8_layer_call_fn_7366657

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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_73642932
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
�
N
2__inference_max_pooling2d_26_layer_call_fn_7364259

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
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_73642532
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
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366564

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
�
�
*__inference_dense_25_layer_call_fn_7366871

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
E__inference_dense_25_layer_call_and_return_conditional_losses_73644252
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
�
__inference_loss_fn_1_7366920O
:dense_24_kernel_regularizer_square_readvariableop_resource:���
identity��1dense_24/kernel/Regularizer/Square/ReadVariableOp�
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_24_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
IdentityIdentity#dense_24/kernel/Regularizer/mul:z:02^dense_24/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp
�
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_7364376

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
E__inference_dense_26_layer_call_and_return_conditional_losses_7366511

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

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
�
%__inference_CNN_layer_call_fn_7365865
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

unknown_13:	�

unknown_14:
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
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_73650392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�
N
2__inference_max_pooling2d_25_layer_call_fn_7364247

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
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_73642412
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
�2
�
@__inference_CNN_layer_call_and_return_conditional_losses_7365179

inputs"
sequential_8_7365126:"
sequential_8_7365128:"
sequential_8_7365130:"
sequential_8_7365132:.
sequential_8_7365134: "
sequential_8_7365136: /
sequential_8_7365138: �#
sequential_8_7365140:	�0
sequential_8_7365142:��#
sequential_8_7365144:	�)
sequential_8_7365146:���#
sequential_8_7365148:	�(
sequential_8_7365150:
��#
sequential_8_7365152:	�#
dense_26_7365155:	�
dense_26_7365157:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp� dense_26/StatefulPartitionedCall�$sequential_8/StatefulPartitionedCall�
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinputssequential_8_7365126sequential_8_7365128sequential_8_7365130sequential_8_7365132sequential_8_7365134sequential_8_7365136sequential_8_7365138sequential_8_7365140sequential_8_7365142sequential_8_7365144sequential_8_7365146sequential_8_7365148sequential_8_7365150sequential_8_7365152*
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_73647752&
$sequential_8/StatefulPartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0dense_26_7365155dense_26_7365157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_73650142"
 dense_26/StatefulPartitionedCall�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_8_7365134*&
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_8_7365146*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_8_7365150* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentity)dense_26/StatefulPartitionedCall:output:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�u
�
__inference_call_7254988

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	�Q
5sequential_8_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	�I
4sequential_8_dense_24_matmul_readvariableop_resource:���D
5sequential_8_dense_24_biasadd_readvariableop_resource:	�H
4sequential_8_dense_25_matmul_readvariableop_resource:
��D
5sequential_8_dense_25_biasadd_readvariableop_resource:	�:
'dense_26_matmul_readvariableop_resource:	�6
(dense_26_biasadd_readvariableop_resource:
identity��dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_8/batch_normalization_8/ReadVariableOp�3sequential_8/batch_normalization_8/ReadVariableOp_1�-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�,sequential_8/conv2d_24/Conv2D/ReadVariableOp�-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�,sequential_8/conv2d_25/Conv2D/ReadVariableOp�-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�,sequential_8/conv2d_26/Conv2D/ReadVariableOp�,sequential_8/dense_24/BiasAdd/ReadVariableOp�+sequential_8/dense_24/MatMul/ReadVariableOp�,sequential_8/dense_25/BiasAdd/ReadVariableOp�+sequential_8/dense_25/MatMul/ReadVariableOp�
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_8/lambda_8/strided_slice/stack�
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_8/lambda_8/strided_slice/stack_1�
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_8/lambda_8/strided_slice/stack_2�
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_8/lambda_8/strided_slice�
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp�
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3�
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_24/Conv2D/ReadVariableOp�
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_8/conv2d_24/Conv2D�
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_8/conv2d_24/BiasAdd�
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_8/conv2d_24/Relu�
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_24/MaxPool�
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_8/conv2d_25/Conv2D/ReadVariableOp�
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_8/conv2d_25/Conv2D�
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_8/conv2d_25/BiasAdd�
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_8/conv2d_25/Relu�
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_25/MaxPool�
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_8/conv2d_26/Conv2D/ReadVariableOp�
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_8/conv2d_26/Conv2D�
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_8/conv2d_26/BiasAdd�
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_8/conv2d_26/Relu�
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_26/MaxPool�
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_8/dropout_24/Identity�
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_8/flatten_8/Const�
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_8/flatten_8/Reshape�
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_8/dense_24/MatMul/ReadVariableOp�
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/MatMul�
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_24/BiasAdd/ReadVariableOp�
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/BiasAdd�
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/Relu�
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_8/dropout_25/Identity�
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_8/dense_25/MatMul/ReadVariableOp�
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/MatMul�
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_25/BiasAdd/ReadVariableOp�
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/BiasAdd�
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/Relu�
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_8/dropout_26/Identity�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/BiasAdd|
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_26/Softmax�
IdentityIdentitydense_26/Softmax:softmax:0 ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366618

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
+__inference_conv2d_25_layer_call_fn_7366722

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
F__inference_conv2d_25_layer_call_and_return_conditional_losses_73643382
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
�
a
E__inference_lambda_8_layer_call_and_return_conditional_losses_7366536

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
�
E__inference_dense_25_layer_call_and_return_conditional_losses_7364425

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_8_layer_call_fn_7366631

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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_73641192
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
�\
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_7364457

inputs+
batch_normalization_8_7364294:+
batch_normalization_8_7364296:+
batch_normalization_8_7364298:+
batch_normalization_8_7364300:+
conv2d_24_7364321: 
conv2d_24_7364323: ,
conv2d_25_7364339: � 
conv2d_25_7364341:	�-
conv2d_26_7364357:�� 
conv2d_26_7364359:	�%
dense_24_7364396:���
dense_24_7364398:	�$
dense_25_7364426:
��
dense_25_7364428:	�
identity��-batch_normalization_8/StatefulPartitionedCall�!conv2d_24/StatefulPartitionedCall�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�!conv2d_25/StatefulPartitionedCall�!conv2d_26/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�1dense_24/kernel/Regularizer/Square/ReadVariableOp� dense_25/StatefulPartitionedCall�1dense_25/kernel/Regularizer/Square/ReadVariableOp�
lambda_8/PartitionedCallPartitionedCallinputs*
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
E__inference_lambda_8_layer_call_and_return_conditional_losses_73642742
lambda_8/PartitionedCall�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0batch_normalization_8_7364294batch_normalization_8_7364296batch_normalization_8_7364298batch_normalization_8_7364300*
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_73642932/
-batch_normalization_8/StatefulPartitionedCall�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_24_7364321conv2d_24_7364323*
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
F__inference_conv2d_24_layer_call_and_return_conditional_losses_73643202#
!conv2d_24/StatefulPartitionedCall�
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_73642292"
 max_pooling2d_24/PartitionedCall�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_7364339conv2d_25_7364341*
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
F__inference_conv2d_25_layer_call_and_return_conditional_losses_73643382#
!conv2d_25/StatefulPartitionedCall�
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_73642412"
 max_pooling2d_25/PartitionedCall�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_7364357conv2d_26_7364359*
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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_73643562#
!conv2d_26/StatefulPartitionedCall�
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_73642532"
 max_pooling2d_26/PartitionedCall�
dropout_24/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
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
G__inference_dropout_24_layer_call_and_return_conditional_losses_73643682
dropout_24/PartitionedCall�
flatten_8/PartitionedCallPartitionedCall#dropout_24/PartitionedCall:output:0*
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
F__inference_flatten_8_layer_call_and_return_conditional_losses_73643762
flatten_8/PartitionedCall�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_24_7364396dense_24_7364398*
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
E__inference_dense_24_layer_call_and_return_conditional_losses_73643952"
 dense_24/StatefulPartitionedCall�
dropout_25/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
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
G__inference_dropout_25_layer_call_and_return_conditional_losses_73644062
dropout_25/PartitionedCall�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_25_7364426dense_25_7364428*
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
E__inference_dense_25_layer_call_and_return_conditional_losses_73644252"
 dense_25/StatefulPartitionedCall�
dropout_26/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
G__inference_dropout_26_layer_call_and_return_conditional_losses_73644362
dropout_26/PartitionedCall�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_24_7364321*&
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_24_7364396*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_7364426* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentity#dropout_26/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_7364241

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
a
E__inference_lambda_8_layer_call_and_return_conditional_losses_7364673

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
G__inference_dropout_24_layer_call_and_return_conditional_losses_7364368

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
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7364119

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
�
e
G__inference_dropout_25_layer_call_and_return_conditional_losses_7366817

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
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366181

inputs;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: C
(conv2d_25_conv2d_readvariableop_resource: �8
)conv2d_25_biasadd_readvariableop_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�<
'dense_24_matmul_readvariableop_resource:���7
(dense_24_biasadd_readvariableop_resource:	�;
'dense_25_matmul_readvariableop_resource:
��7
(dense_25_biasadd_readvariableop_resource:	�
identity��$batch_normalization_8/AssignNewValue�&batch_normalization_8/AssignNewValue_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�
lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_8/strided_slice/stack�
lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_8/strided_slice/stack_1�
lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_8/strided_slice/stack_2�
lambda_8/strided_sliceStridedSliceinputs%lambda_8/strided_slice/stack:output:0'lambda_8/strided_slice/stack_1:output:0'lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_8/strided_slice�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3lambda_8/strided_slice:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_8/FusedBatchNormV3�
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue�
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
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
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_24/MaxPool�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_25/Conv2D/ReadVariableOp�
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
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
:���������%%�2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_25/Relu�
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_25/MaxPool�
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_26/Conv2D/ReadVariableOp�
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
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
:����������2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_26/Relu�
max_pooling2d_26/MaxPoolMaxPoolconv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPooly
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_24/dropout/Const�
dropout_24/dropout/MulMul!max_pooling2d_26/MaxPool:output:0!dropout_24/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2
dropout_24/dropout/Mul�
dropout_24/dropout/ShapeShape!max_pooling2d_26/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_24/dropout/Shape�
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype021
/dropout_24/dropout/random_uniform/RandomUniform�
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_24/dropout/GreaterEqual/y�
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2!
dropout_24/dropout/GreaterEqual�
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout_24/dropout/Cast�
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout_24/dropout/Mul_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_8/Const�
flatten_8/ReshapeReshapedropout_24/dropout/Mul_1:z:0flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_8/Reshape�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02 
dense_24/MatMul/ReadVariableOp�
dense_24/MatMulMatMulflatten_8/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_24/MatMul�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_24/BiasAdd/ReadVariableOp�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_24/BiasAddt
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_24/Reluy
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_25/dropout/Const�
dropout_25/dropout/MulMuldense_24/Relu:activations:0!dropout_25/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_25/dropout/Mul
dropout_25/dropout/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:2
dropout_25/dropout/Shape�
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_25/dropout/random_uniform/RandomUniform�
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_25/dropout/GreaterEqual/y�
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_25/dropout/GreaterEqual�
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_25/dropout/Cast�
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_25/dropout/Mul_1�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_25/MatMul/ReadVariableOp�
dense_25/MatMulMatMuldropout_25/dropout/Mul_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_25/MatMul�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_25/BiasAdd/ReadVariableOp�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_25/BiasAddt
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_25/Reluy
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_26/dropout/Const�
dropout_26/dropout/MulMuldense_25/Relu:activations:0!dropout_26/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_26/dropout/Mul
dropout_26/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
dropout_26/dropout/Shape�
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform�
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_26/dropout/GreaterEqual/y�
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_26/dropout/GreaterEqual�
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_26/dropout/Cast�
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_26/dropout/Mul_1�
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentitydropout_26/dropout/Mul_1:z:0%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7364293

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
�
F__inference_conv2d_25_layer_call_and_return_conditional_losses_7364338

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
E__inference_dense_25_layer_call_and_return_conditional_losses_7366862

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_24_layer_call_and_return_conditional_losses_7364320

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
�
e
,__inference_dropout_25_layer_call_fn_7366839

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
G__inference_dropout_25_layer_call_and_return_conditional_losses_73645412
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
�
e
G__inference_dropout_26_layer_call_and_return_conditional_losses_7366876

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
�v
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366077

inputs;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: C
(conv2d_25_conv2d_readvariableop_resource: �8
)conv2d_25_biasadd_readvariableop_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�<
'dense_24_matmul_readvariableop_resource:���7
(dense_24_biasadd_readvariableop_resource:	�;
'dense_25_matmul_readvariableop_resource:
��7
(dense_25_biasadd_readvariableop_resource:	�
identity��5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�
lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_8/strided_slice/stack�
lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_8/strided_slice/stack_1�
lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_8/strided_slice/stack_2�
lambda_8/strided_sliceStridedSliceinputs%lambda_8/strided_slice/stack:output:0'lambda_8/strided_slice/stack_1:output:0'lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_8/strided_slice�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3lambda_8/strided_slice:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
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
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_24/MaxPool�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_25/Conv2D/ReadVariableOp�
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
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
:���������%%�2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_25/Relu�
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_25/MaxPool�
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_26/Conv2D/ReadVariableOp�
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
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
:����������2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_26/Relu�
max_pooling2d_26/MaxPoolMaxPoolconv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPool�
dropout_24/IdentityIdentity!max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2
dropout_24/Identitys
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_8/Const�
flatten_8/ReshapeReshapedropout_24/Identity:output:0flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_8/Reshape�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02 
dense_24/MatMul/ReadVariableOp�
dense_24/MatMulMatMulflatten_8/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_24/MatMul�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_24/BiasAdd/ReadVariableOp�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_24/BiasAddt
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_24/Relu�
dropout_25/IdentityIdentitydense_24/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_25/Identity�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_25/MatMul/ReadVariableOp�
dense_25/MatMulMatMuldropout_25/Identity:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_25/MatMul�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_25/BiasAdd/ReadVariableOp�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_25/BiasAddt
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_25/Relu�
dropout_26/IdentityIdentitydense_25/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_26/Identity�
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentitydropout_26/Identity:output:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_7366775

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
�
e
G__inference_dropout_24_layer_call_and_return_conditional_losses_7366747

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
�
.__inference_sequential_8_layer_call_fn_7366467

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
I__inference_sequential_8_layer_call_and_return_conditional_losses_73647752
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
�v
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366264
lambda_8_input;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: C
(conv2d_25_conv2d_readvariableop_resource: �8
)conv2d_25_biasadd_readvariableop_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�<
'dense_24_matmul_readvariableop_resource:���7
(dense_24_biasadd_readvariableop_resource:	�;
'dense_25_matmul_readvariableop_resource:
��7
(dense_25_biasadd_readvariableop_resource:	�
identity��5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�
lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_8/strided_slice/stack�
lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_8/strided_slice/stack_1�
lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_8/strided_slice/stack_2�
lambda_8/strided_sliceStridedSlicelambda_8_input%lambda_8/strided_slice/stack:output:0'lambda_8/strided_slice/stack_1:output:0'lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_8/strided_slice�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3lambda_8/strided_slice:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
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
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_24/MaxPool�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_25/Conv2D/ReadVariableOp�
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
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
:���������%%�2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_25/Relu�
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_25/MaxPool�
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_26/Conv2D/ReadVariableOp�
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
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
:����������2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_26/Relu�
max_pooling2d_26/MaxPoolMaxPoolconv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPool�
dropout_24/IdentityIdentity!max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2
dropout_24/Identitys
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_8/Const�
flatten_8/ReshapeReshapedropout_24/Identity:output:0flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_8/Reshape�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02 
dense_24/MatMul/ReadVariableOp�
dense_24/MatMulMatMulflatten_8/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_24/MatMul�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_24/BiasAdd/ReadVariableOp�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_24/BiasAddt
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_24/Relu�
dropout_25/IdentityIdentitydense_24/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_25/Identity�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_25/MatMul/ReadVariableOp�
dense_25/MatMulMatMuldropout_25/Identity:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_25/MatMul�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_25/BiasAdd/ReadVariableOp�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_25/BiasAddt
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_25/Relu�
dropout_26/IdentityIdentitydense_25/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_26/Identity�
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentitydropout_26/Identity:output:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_8_input
��
�!
#__inference__traced_restore_7367282
file_prefix3
 assignvariableop_dense_26_kernel:	�.
 assignvariableop_1_dense_26_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_8_gamma:;
-assignvariableop_8_batch_normalization_8_beta:B
4assignvariableop_9_batch_normalization_8_moving_mean:G
9assignvariableop_10_batch_normalization_8_moving_variance:>
$assignvariableop_11_conv2d_24_kernel: 0
"assignvariableop_12_conv2d_24_bias: ?
$assignvariableop_13_conv2d_25_kernel: �1
"assignvariableop_14_conv2d_25_bias:	�@
$assignvariableop_15_conv2d_26_kernel:��1
"assignvariableop_16_conv2d_26_bias:	�8
#assignvariableop_17_dense_24_kernel:���0
!assignvariableop_18_dense_24_bias:	�7
#assignvariableop_19_dense_25_kernel:
��0
!assignvariableop_20_dense_25_bias:	�#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_26_kernel_m:	�6
(assignvariableop_26_adam_dense_26_bias_m:D
6assignvariableop_27_adam_batch_normalization_8_gamma_m:C
5assignvariableop_28_adam_batch_normalization_8_beta_m:E
+assignvariableop_29_adam_conv2d_24_kernel_m: 7
)assignvariableop_30_adam_conv2d_24_bias_m: F
+assignvariableop_31_adam_conv2d_25_kernel_m: �8
)assignvariableop_32_adam_conv2d_25_bias_m:	�G
+assignvariableop_33_adam_conv2d_26_kernel_m:��8
)assignvariableop_34_adam_conv2d_26_bias_m:	�?
*assignvariableop_35_adam_dense_24_kernel_m:���7
(assignvariableop_36_adam_dense_24_bias_m:	�>
*assignvariableop_37_adam_dense_25_kernel_m:
��7
(assignvariableop_38_adam_dense_25_bias_m:	�=
*assignvariableop_39_adam_dense_26_kernel_v:	�6
(assignvariableop_40_adam_dense_26_bias_v:D
6assignvariableop_41_adam_batch_normalization_8_gamma_v:C
5assignvariableop_42_adam_batch_normalization_8_beta_v:E
+assignvariableop_43_adam_conv2d_24_kernel_v: 7
)assignvariableop_44_adam_conv2d_24_bias_v: F
+assignvariableop_45_adam_conv2d_25_kernel_v: �8
)assignvariableop_46_adam_conv2d_25_bias_v:	�G
+assignvariableop_47_adam_conv2d_26_kernel_v:��8
)assignvariableop_48_adam_conv2d_26_bias_v:	�?
*assignvariableop_49_adam_dense_24_kernel_v:���7
(assignvariableop_50_adam_dense_24_bias_v:	�>
*assignvariableop_51_adam_dense_25_kernel_v:
��7
(assignvariableop_52_adam_dense_25_bias_v:	�
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
AssignVariableOpAssignVariableOp assignvariableop_dense_26_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_26_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_8_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_8_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_8_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_8_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_24_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_24_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_26_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_26_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_24_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_24_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_25_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_25_biasIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_26_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_26_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_8_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_8_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_24_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_24_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_25_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_25_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_26_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_26_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_24_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_24_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_25_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_25_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_26_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_26_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_8_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_8_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_24_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_24_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_25_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_25_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_26_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_26_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_24_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_24_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_25_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_25_bias_vIdentity_52:output:0"/device:CPU:0*
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
�
a
E__inference_lambda_8_layer_call_and_return_conditional_losses_7364274

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
H
,__inference_dropout_24_layer_call_fn_7366764

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
G__inference_dropout_24_layer_call_and_return_conditional_losses_73643682
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
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7364163

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
�
�
*__inference_dense_24_layer_call_fn_7366812

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
E__inference_dense_24_layer_call_and_return_conditional_losses_73643952
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
��
�
@__inference_CNN_layer_call_and_return_conditional_losses_7365828
input_1H
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	�Q
5sequential_8_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	�I
4sequential_8_dense_24_matmul_readvariableop_resource:���D
5sequential_8_dense_24_biasadd_readvariableop_resource:	�H
4sequential_8_dense_25_matmul_readvariableop_resource:
��D
5sequential_8_dense_25_biasadd_readvariableop_resource:	�:
'dense_26_matmul_readvariableop_resource:	�6
(dense_26_biasadd_readvariableop_resource:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�1sequential_8/batch_normalization_8/AssignNewValue�3sequential_8/batch_normalization_8/AssignNewValue_1�Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_8/batch_normalization_8/ReadVariableOp�3sequential_8/batch_normalization_8/ReadVariableOp_1�-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�,sequential_8/conv2d_24/Conv2D/ReadVariableOp�-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�,sequential_8/conv2d_25/Conv2D/ReadVariableOp�-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�,sequential_8/conv2d_26/Conv2D/ReadVariableOp�,sequential_8/dense_24/BiasAdd/ReadVariableOp�+sequential_8/dense_24/MatMul/ReadVariableOp�,sequential_8/dense_25/BiasAdd/ReadVariableOp�+sequential_8/dense_25/MatMul/ReadVariableOp�
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_8/lambda_8/strided_slice/stack�
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_8/lambda_8/strided_slice/stack_1�
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_8/lambda_8/strided_slice/stack_2�
#sequential_8/lambda_8/strided_sliceStridedSliceinput_12sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_8/lambda_8/strided_slice�
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp�
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<25
3sequential_8/batch_normalization_8/FusedBatchNormV3�
1sequential_8/batch_normalization_8/AssignNewValueAssignVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource@sequential_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_8/batch_normalization_8/AssignNewValue�
3sequential_8/batch_normalization_8/AssignNewValue_1AssignVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceDsequential_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0E^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_8/batch_normalization_8/AssignNewValue_1�
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_24/Conv2D/ReadVariableOp�
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_8/conv2d_24/Conv2D�
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_8/conv2d_24/BiasAdd�
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_8/conv2d_24/Relu�
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_24/MaxPool�
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_8/conv2d_25/Conv2D/ReadVariableOp�
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_8/conv2d_25/Conv2D�
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_8/conv2d_25/BiasAdd�
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_8/conv2d_25/Relu�
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_25/MaxPool�
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_8/conv2d_26/Conv2D/ReadVariableOp�
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_8/conv2d_26/Conv2D�
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_8/conv2d_26/BiasAdd�
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_8/conv2d_26/Relu�
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_26/MaxPool�
%sequential_8/dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2'
%sequential_8/dropout_24/dropout/Const�
#sequential_8/dropout_24/dropout/MulMul.sequential_8/max_pooling2d_26/MaxPool:output:0.sequential_8/dropout_24/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2%
#sequential_8/dropout_24/dropout/Mul�
%sequential_8/dropout_24/dropout/ShapeShape.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_8/dropout_24/dropout/Shape�
<sequential_8/dropout_24/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_24/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype02>
<sequential_8/dropout_24/dropout/random_uniform/RandomUniform�
.sequential_8/dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=20
.sequential_8/dropout_24/dropout/GreaterEqual/y�
,sequential_8/dropout_24/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_24/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_24/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2.
,sequential_8/dropout_24/dropout/GreaterEqual�
$sequential_8/dropout_24/dropout/CastCast0sequential_8/dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2&
$sequential_8/dropout_24/dropout/Cast�
%sequential_8/dropout_24/dropout/Mul_1Mul'sequential_8/dropout_24/dropout/Mul:z:0(sequential_8/dropout_24/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2'
%sequential_8/dropout_24/dropout/Mul_1�
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_8/flatten_8/Const�
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/dropout/Mul_1:z:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_8/flatten_8/Reshape�
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_8/dense_24/MatMul/ReadVariableOp�
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/MatMul�
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_24/BiasAdd/ReadVariableOp�
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/BiasAdd�
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/Relu�
%sequential_8/dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_8/dropout_25/dropout/Const�
#sequential_8/dropout_25/dropout/MulMul(sequential_8/dense_24/Relu:activations:0.sequential_8/dropout_25/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_8/dropout_25/dropout/Mul�
%sequential_8/dropout_25/dropout/ShapeShape(sequential_8/dense_24/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_8/dropout_25/dropout/Shape�
<sequential_8/dropout_25/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_25/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_8/dropout_25/dropout/random_uniform/RandomUniform�
.sequential_8/dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_8/dropout_25/dropout/GreaterEqual/y�
,sequential_8/dropout_25/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_25/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_25/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_8/dropout_25/dropout/GreaterEqual�
$sequential_8/dropout_25/dropout/CastCast0sequential_8/dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_8/dropout_25/dropout/Cast�
%sequential_8/dropout_25/dropout/Mul_1Mul'sequential_8/dropout_25/dropout/Mul:z:0(sequential_8/dropout_25/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_8/dropout_25/dropout/Mul_1�
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_8/dense_25/MatMul/ReadVariableOp�
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/dropout/Mul_1:z:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/MatMul�
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_25/BiasAdd/ReadVariableOp�
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/BiasAdd�
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/Relu�
%sequential_8/dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_8/dropout_26/dropout/Const�
#sequential_8/dropout_26/dropout/MulMul(sequential_8/dense_25/Relu:activations:0.sequential_8/dropout_26/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_8/dropout_26/dropout/Mul�
%sequential_8/dropout_26/dropout/ShapeShape(sequential_8/dense_25/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_8/dropout_26/dropout/Shape�
<sequential_8/dropout_26/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_26/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_8/dropout_26/dropout/random_uniform/RandomUniform�
.sequential_8/dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_8/dropout_26/dropout/GreaterEqual/y�
,sequential_8/dropout_26/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_26/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_26/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_8/dropout_26/dropout/GreaterEqual�
$sequential_8/dropout_26/dropout/CastCast0sequential_8/dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_8/dropout_26/dropout/Cast�
%sequential_8/dropout_26/dropout/Mul_1Mul'sequential_8/dropout_26/dropout/Mul:z:0(sequential_8/dropout_26/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_8/dropout_26/dropout/Mul_1�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMul)sequential_8/dropout_26/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/BiasAdd|
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_26/Softmax�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�	
IdentityIdentitydense_26/Softmax:softmax:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp2^sequential_8/batch_normalization_8/AssignNewValue4^sequential_8/batch_normalization_8/AssignNewValue_1C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2f
1sequential_8/batch_normalization_8/AssignNewValue1sequential_8/batch_normalization_8/AssignNewValue2j
3sequential_8/batch_normalization_8/AssignNewValue_13sequential_8/batch_normalization_8/AssignNewValue_12�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7364646

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
�
.__inference_sequential_8_layer_call_fn_7366401
lambda_8_input
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
StatefulPartitionedCallStatefulPartitionedCalllambda_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_73644572
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
_user_specified_namelambda_8_input
�u
�
__inference_call_7257118

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	�Q
5sequential_8_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	�I
4sequential_8_dense_24_matmul_readvariableop_resource:���D
5sequential_8_dense_24_biasadd_readvariableop_resource:	�H
4sequential_8_dense_25_matmul_readvariableop_resource:
��D
5sequential_8_dense_25_biasadd_readvariableop_resource:	�:
'dense_26_matmul_readvariableop_resource:	�6
(dense_26_biasadd_readvariableop_resource:
identity��dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_8/batch_normalization_8/ReadVariableOp�3sequential_8/batch_normalization_8/ReadVariableOp_1�-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�,sequential_8/conv2d_24/Conv2D/ReadVariableOp�-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�,sequential_8/conv2d_25/Conv2D/ReadVariableOp�-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�,sequential_8/conv2d_26/Conv2D/ReadVariableOp�,sequential_8/dense_24/BiasAdd/ReadVariableOp�+sequential_8/dense_24/MatMul/ReadVariableOp�,sequential_8/dense_25/BiasAdd/ReadVariableOp�+sequential_8/dense_25/MatMul/ReadVariableOp�
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_8/lambda_8/strided_slice/stack�
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_8/lambda_8/strided_slice/stack_1�
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_8/lambda_8/strided_slice/stack_2�
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_8/lambda_8/strided_slice�
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp�
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3�
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_24/Conv2D/ReadVariableOp�
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_8/conv2d_24/Conv2D�
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_8/conv2d_24/BiasAdd�
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_8/conv2d_24/Relu�
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_24/MaxPool�
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_8/conv2d_25/Conv2D/ReadVariableOp�
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_8/conv2d_25/Conv2D�
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_8/conv2d_25/BiasAdd�
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_8/conv2d_25/Relu�
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_25/MaxPool�
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_8/conv2d_26/Conv2D/ReadVariableOp�
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_8/conv2d_26/Conv2D�
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_8/conv2d_26/BiasAdd�
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_8/conv2d_26/Relu�
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_26/MaxPool�
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_8/dropout_24/Identity�
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_8/flatten_8/Const�
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_8/flatten_8/Reshape�
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_8/dense_24/MatMul/ReadVariableOp�
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/MatMul�
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_24/BiasAdd/ReadVariableOp�
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/BiasAdd�
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/Relu�
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_8/dropout_25/Identity�
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_8/dense_25/MatMul/ReadVariableOp�
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/MatMul�
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_25/BiasAdd/ReadVariableOp�
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/BiasAdd�
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/Relu�
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_8/dropout_26/Identity�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/BiasAdd|
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_26/Softmax�
IdentityIdentitydense_26/Softmax:softmax:0 ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
a
E__inference_lambda_8_layer_call_and_return_conditional_losses_7366528

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
+__inference_conv2d_26_layer_call_fn_7366742

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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_73643562
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
,__inference_dropout_26_layer_call_fn_7366893

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
G__inference_dropout_26_layer_call_and_return_conditional_losses_73644362
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
�h
�
 __inference__traced_save_7367113
file_prefix.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableop6
2savev2_adam_conv2d_25_kernel_m_read_readvariableop4
0savev2_adam_conv2d_25_bias_m_read_readvariableop6
2savev2_adam_conv2d_26_kernel_m_read_readvariableop4
0savev2_adam_conv2d_26_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_v_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableop6
2savev2_adam_conv2d_25_kernel_v_read_readvariableop4
0savev2_adam_conv2d_25_bias_v_read_readvariableop6
2savev2_adam_conv2d_26_kernel_v_read_readvariableop4
0savev2_adam_conv2d_26_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop2savev2_adam_conv2d_25_kernel_m_read_readvariableop0savev2_adam_conv2d_25_bias_m_read_readvariableop2savev2_adam_conv2d_26_kernel_m_read_readvariableop0savev2_adam_conv2d_26_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop2savev2_adam_conv2d_25_kernel_v_read_readvariableop0savev2_adam_conv2d_25_bias_v_read_readvariableop2savev2_adam_conv2d_26_kernel_v_read_readvariableop0savev2_adam_conv2d_26_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�: :	�:: : : : : ::::: : : �:�:��:�:���:�:
��:�: : : : :	�:::: : : �:�:��:�:���:�:
��:�:	�:::: : : �:�:��:�:���:�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�: 

_output_shapes
::
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
:	�: 

_output_shapes
:: 
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
:	�: )

_output_shapes
:: *
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
��
�
@__inference_CNN_layer_call_and_return_conditional_losses_7365516

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	�Q
5sequential_8_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	�I
4sequential_8_dense_24_matmul_readvariableop_resource:���D
5sequential_8_dense_24_biasadd_readvariableop_resource:	�H
4sequential_8_dense_25_matmul_readvariableop_resource:
��D
5sequential_8_dense_25_biasadd_readvariableop_resource:	�:
'dense_26_matmul_readvariableop_resource:	�6
(dense_26_biasadd_readvariableop_resource:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_8/batch_normalization_8/ReadVariableOp�3sequential_8/batch_normalization_8/ReadVariableOp_1�-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�,sequential_8/conv2d_24/Conv2D/ReadVariableOp�-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�,sequential_8/conv2d_25/Conv2D/ReadVariableOp�-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�,sequential_8/conv2d_26/Conv2D/ReadVariableOp�,sequential_8/dense_24/BiasAdd/ReadVariableOp�+sequential_8/dense_24/MatMul/ReadVariableOp�,sequential_8/dense_25/BiasAdd/ReadVariableOp�+sequential_8/dense_25/MatMul/ReadVariableOp�
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_8/lambda_8/strided_slice/stack�
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_8/lambda_8/strided_slice/stack_1�
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_8/lambda_8/strided_slice/stack_2�
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_8/lambda_8/strided_slice�
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp�
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3�
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_24/Conv2D/ReadVariableOp�
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_8/conv2d_24/Conv2D�
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_8/conv2d_24/BiasAdd�
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_8/conv2d_24/Relu�
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_24/MaxPool�
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_8/conv2d_25/Conv2D/ReadVariableOp�
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_8/conv2d_25/Conv2D�
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_8/conv2d_25/BiasAdd�
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_8/conv2d_25/Relu�
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_25/MaxPool�
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_8/conv2d_26/Conv2D/ReadVariableOp�
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_8/conv2d_26/Conv2D�
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_8/conv2d_26/BiasAdd�
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_8/conv2d_26/Relu�
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_26/MaxPool�
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_8/dropout_24/Identity�
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_8/flatten_8/Const�
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_8/flatten_8/Reshape�
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_8/dense_24/MatMul/ReadVariableOp�
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/MatMul�
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_24/BiasAdd/ReadVariableOp�
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/BiasAdd�
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/Relu�
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_8/dropout_25/Identity�
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_8/dense_25/MatMul/ReadVariableOp�
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/MatMul�
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_25/BiasAdd/ReadVariableOp�
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/BiasAdd�
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/Relu�
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_8/dropout_26/Identity�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/BiasAdd|
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_26/Softmax�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentitydense_26/Softmax:softmax:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
F__inference_conv2d_25_layer_call_and_return_conditional_losses_7366713

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
__inference_loss_fn_2_7366931N
:dense_25_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_25/kernel/Regularizer/Square/ReadVariableOp�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_25_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentity#dense_25/kernel/Regularizer/mul:z:02^dense_25/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp
�
�
%__inference_CNN_layer_call_fn_7365902

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

unknown_13:	�

unknown_14:
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
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_73650392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�
F
*__inference_lambda_8_layer_call_fn_7366541

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
E__inference_lambda_8_layer_call_and_return_conditional_losses_73642742
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
G__inference_dropout_24_layer_call_and_return_conditional_losses_7366759

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
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366600

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
%__inference_CNN_layer_call_fn_7365939

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

unknown_13:	�

unknown_14:
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
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_73651792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�
e
,__inference_dropout_26_layer_call_fn_7366898

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
G__inference_dropout_26_layer_call_and_return_conditional_losses_73645082
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
�
i
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_7364253

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
�
F__inference_conv2d_26_layer_call_and_return_conditional_losses_7364356

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
+__inference_conv2d_24_layer_call_fn_7366702

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
F__inference_conv2d_24_layer_call_and_return_conditional_losses_73643202
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
�
@__inference_CNN_layer_call_and_return_conditional_losses_7365627

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: �E
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	�Q
5sequential_8_conv2d_26_conv2d_readvariableop_resource:��E
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	�I
4sequential_8_dense_24_matmul_readvariableop_resource:���D
5sequential_8_dense_24_biasadd_readvariableop_resource:	�H
4sequential_8_dense_25_matmul_readvariableop_resource:
��D
5sequential_8_dense_25_biasadd_readvariableop_resource:	�:
'dense_26_matmul_readvariableop_resource:	�6
(dense_26_biasadd_readvariableop_resource:
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�1sequential_8/batch_normalization_8/AssignNewValue�3sequential_8/batch_normalization_8/AssignNewValue_1�Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_8/batch_normalization_8/ReadVariableOp�3sequential_8/batch_normalization_8/ReadVariableOp_1�-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�,sequential_8/conv2d_24/Conv2D/ReadVariableOp�-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�,sequential_8/conv2d_25/Conv2D/ReadVariableOp�-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�,sequential_8/conv2d_26/Conv2D/ReadVariableOp�,sequential_8/dense_24/BiasAdd/ReadVariableOp�+sequential_8/dense_24/MatMul/ReadVariableOp�,sequential_8/dense_25/BiasAdd/ReadVariableOp�+sequential_8/dense_25/MatMul/ReadVariableOp�
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_8/lambda_8/strided_slice/stack�
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_8/lambda_8/strided_slice/stack_1�
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_8/lambda_8/strided_slice/stack_2�
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_8/lambda_8/strided_slice�
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp�
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<25
3sequential_8/batch_normalization_8/FusedBatchNormV3�
1sequential_8/batch_normalization_8/AssignNewValueAssignVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource@sequential_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_8/batch_normalization_8/AssignNewValue�
3sequential_8/batch_normalization_8/AssignNewValue_1AssignVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceDsequential_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0E^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_8/batch_normalization_8/AssignNewValue_1�
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_8/conv2d_24/Conv2D/ReadVariableOp�
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_8/conv2d_24/Conv2D�
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp�
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_8/conv2d_24/BiasAdd�
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_8/conv2d_24/Relu�
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_24/MaxPool�
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_8/conv2d_25/Conv2D/ReadVariableOp�
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_8/conv2d_25/Conv2D�
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp�
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_8/conv2d_25/BiasAdd�
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_8/conv2d_25/Relu�
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_25/MaxPool�
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_8/conv2d_26/Conv2D/ReadVariableOp�
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_8/conv2d_26/Conv2D�
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp�
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_8/conv2d_26/BiasAdd�
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_8/conv2d_26/Relu�
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_8/max_pooling2d_26/MaxPool�
%sequential_8/dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2'
%sequential_8/dropout_24/dropout/Const�
#sequential_8/dropout_24/dropout/MulMul.sequential_8/max_pooling2d_26/MaxPool:output:0.sequential_8/dropout_24/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2%
#sequential_8/dropout_24/dropout/Mul�
%sequential_8/dropout_24/dropout/ShapeShape.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_8/dropout_24/dropout/Shape�
<sequential_8/dropout_24/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_24/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype02>
<sequential_8/dropout_24/dropout/random_uniform/RandomUniform�
.sequential_8/dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=20
.sequential_8/dropout_24/dropout/GreaterEqual/y�
,sequential_8/dropout_24/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_24/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_24/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2.
,sequential_8/dropout_24/dropout/GreaterEqual�
$sequential_8/dropout_24/dropout/CastCast0sequential_8/dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2&
$sequential_8/dropout_24/dropout/Cast�
%sequential_8/dropout_24/dropout/Mul_1Mul'sequential_8/dropout_24/dropout/Mul:z:0(sequential_8/dropout_24/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2'
%sequential_8/dropout_24/dropout/Mul_1�
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_8/flatten_8/Const�
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/dropout/Mul_1:z:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_8/flatten_8/Reshape�
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_8/dense_24/MatMul/ReadVariableOp�
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/MatMul�
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_24/BiasAdd/ReadVariableOp�
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/BiasAdd�
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_24/Relu�
%sequential_8/dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_8/dropout_25/dropout/Const�
#sequential_8/dropout_25/dropout/MulMul(sequential_8/dense_24/Relu:activations:0.sequential_8/dropout_25/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_8/dropout_25/dropout/Mul�
%sequential_8/dropout_25/dropout/ShapeShape(sequential_8/dense_24/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_8/dropout_25/dropout/Shape�
<sequential_8/dropout_25/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_25/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_8/dropout_25/dropout/random_uniform/RandomUniform�
.sequential_8/dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_8/dropout_25/dropout/GreaterEqual/y�
,sequential_8/dropout_25/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_25/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_25/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_8/dropout_25/dropout/GreaterEqual�
$sequential_8/dropout_25/dropout/CastCast0sequential_8/dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_8/dropout_25/dropout/Cast�
%sequential_8/dropout_25/dropout/Mul_1Mul'sequential_8/dropout_25/dropout/Mul:z:0(sequential_8/dropout_25/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_8/dropout_25/dropout/Mul_1�
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_8/dense_25/MatMul/ReadVariableOp�
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/dropout/Mul_1:z:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/MatMul�
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_8/dense_25/BiasAdd/ReadVariableOp�
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/BiasAdd�
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_8/dense_25/Relu�
%sequential_8/dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_8/dropout_26/dropout/Const�
#sequential_8/dropout_26/dropout/MulMul(sequential_8/dense_25/Relu:activations:0.sequential_8/dropout_26/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_8/dropout_26/dropout/Mul�
%sequential_8/dropout_26/dropout/ShapeShape(sequential_8/dense_25/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_8/dropout_26/dropout/Shape�
<sequential_8/dropout_26/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_26/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_8/dropout_26/dropout/random_uniform/RandomUniform�
.sequential_8/dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_8/dropout_26/dropout/GreaterEqual/y�
,sequential_8/dropout_26/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_26/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_26/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_8/dropout_26/dropout/GreaterEqual�
$sequential_8/dropout_26/dropout/CastCast0sequential_8/dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_8/dropout_26/dropout/Cast�
%sequential_8/dropout_26/dropout/Mul_1Mul'sequential_8/dropout_26/dropout/Mul:z:0(sequential_8/dropout_26/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_8/dropout_26/dropout/Mul_1�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMul)sequential_8/dropout_26/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/BiasAdd|
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_26/Softmax�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�	
IdentityIdentitydense_26/Softmax:softmax:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp2^sequential_8/batch_normalization_8/AssignNewValue4^sequential_8/batch_normalization_8/AssignNewValue_1C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2f
1sequential_8/batch_normalization_8/AssignNewValue1sequential_8/batch_normalization_8/AssignNewValue2j
3sequential_8/batch_normalization_8/AssignNewValue_13sequential_8/batch_normalization_8/AssignNewValue_12�
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
Қ
�
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366368
lambda_8_input;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: C
(conv2d_25_conv2d_readvariableop_resource: �8
)conv2d_25_biasadd_readvariableop_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�<
'dense_24_matmul_readvariableop_resource:���7
(dense_24_biasadd_readvariableop_resource:	�;
'dense_25_matmul_readvariableop_resource:
��7
(dense_25_biasadd_readvariableop_resource:	�
identity��$batch_normalization_8/AssignNewValue�&batch_normalization_8/AssignNewValue_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�1dense_25/kernel/Regularizer/Square/ReadVariableOp�
lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_8/strided_slice/stack�
lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_8/strided_slice/stack_1�
lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_8/strided_slice/stack_2�
lambda_8/strided_sliceStridedSlicelambda_8_input%lambda_8/strided_slice/stack:output:0'lambda_8/strided_slice/stack_1:output:0'lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_8/strided_slice�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3lambda_8/strided_slice:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_8/FusedBatchNormV3�
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue�
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
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
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_24/MaxPool�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_25/Conv2D/ReadVariableOp�
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
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
:���������%%�2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_25/Relu�
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_25/MaxPool�
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_26/Conv2D/ReadVariableOp�
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
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
:����������2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_26/Relu�
max_pooling2d_26/MaxPoolMaxPoolconv2d_26/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPooly
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_24/dropout/Const�
dropout_24/dropout/MulMul!max_pooling2d_26/MaxPool:output:0!dropout_24/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2
dropout_24/dropout/Mul�
dropout_24/dropout/ShapeShape!max_pooling2d_26/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_24/dropout/Shape�
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype021
/dropout_24/dropout/random_uniform/RandomUniform�
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_24/dropout/GreaterEqual/y�
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2!
dropout_24/dropout/GreaterEqual�
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout_24/dropout/Cast�
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout_24/dropout/Mul_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_8/Const�
flatten_8/ReshapeReshapedropout_24/dropout/Mul_1:z:0flatten_8/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_8/Reshape�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02 
dense_24/MatMul/ReadVariableOp�
dense_24/MatMulMatMulflatten_8/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_24/MatMul�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_24/BiasAdd/ReadVariableOp�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_24/BiasAddt
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_24/Reluy
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_25/dropout/Const�
dropout_25/dropout/MulMuldense_24/Relu:activations:0!dropout_25/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_25/dropout/Mul
dropout_25/dropout/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:2
dropout_25/dropout/Shape�
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_25/dropout/random_uniform/RandomUniform�
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_25/dropout/GreaterEqual/y�
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_25/dropout/GreaterEqual�
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_25/dropout/Cast�
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_25/dropout/Mul_1�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_25/MatMul/ReadVariableOp�
dense_25/MatMulMatMuldropout_25/dropout/Mul_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_25/MatMul�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_25/BiasAdd/ReadVariableOp�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_25/BiasAddt
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_25/Reluy
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_26/dropout/Const�
dropout_26/dropout/MulMuldense_25/Relu:activations:0!dropout_26/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_26/dropout/Mul
dropout_26/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
dropout_26/dropout/Shape�
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform�
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_26/dropout/GreaterEqual/y�
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_26/dropout/GreaterEqual�
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_26/dropout/Cast�
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_26/dropout/Mul_1�
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_25/kernel/Regularizer/Square/ReadVariableOp�
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_25/kernel/Regularizer/Square�
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_25/kernel/Regularizer/Const�
dense_25/kernel/Regularizer/SumSum&dense_25/kernel/Regularizer/Square:y:0*dense_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/Sum�
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_25/kernel/Regularizer/mul/x�
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_25/kernel/Regularizer/mul�
IdentityIdentitydropout_26/dropout/Mul_1:z:0%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_8_input
�

�
E__inference_dense_26_layer_call_and_return_conditional_losses_7365014

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

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
�
"__inference__wrapped_model_7364097
input_1
cnn_7364063:
cnn_7364065:
cnn_7364067:
cnn_7364069:%
cnn_7364071: 
cnn_7364073: &
cnn_7364075: �
cnn_7364077:	�'
cnn_7364079:��
cnn_7364081:	� 
cnn_7364083:���
cnn_7364085:	�
cnn_7364087:
��
cnn_7364089:	�
cnn_7364091:	�
cnn_7364093:
identity��CNN/StatefulPartitionedCall�
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn_7364063cnn_7364065cnn_7364067cnn_7364069cnn_7364071cnn_7364073cnn_7364075cnn_7364077cnn_7364079cnn_7364081cnn_7364083cnn_7364085cnn_7364087cnn_7364089cnn_7364091cnn_7364093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *!
fR
__inference_call_72549882
CNN/StatefulPartitionedCall�
IdentityIdentity$CNN/StatefulPartitionedCall:output:0^CNN/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�
i
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_7364229

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
�
�
*__inference_dense_26_layer_call_fn_7366520

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_73650142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�
7__inference_batch_normalization_8_layer_call_fn_7366670

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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_73646462
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
�
�
F__inference_conv2d_24_layer_call_and_return_conditional_losses_7366693

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
�
e
G__inference_dropout_25_layer_call_and_return_conditional_losses_7364406

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
E__inference_dense_24_layer_call_and_return_conditional_losses_7364395

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_24/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp�
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_24/kernel/Regularizer/Square�
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const�
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum�
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_24/kernel/Regularizer/mul/x�
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
f
G__inference_dropout_25_layer_call_and_return_conditional_losses_7364541

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
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�


h2ptjl
_output
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
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
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�d
_tf_keras_sequential�d{"name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_8_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_8", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4QAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_25", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_8_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_8_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_8", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4QAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_25", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem�m�&m�'m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�v�v�&v�'v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�"
	optimizer
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
 "
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
	variables
regularization_losses
trainable_variables
4metrics
5layer_regularization_losses
6layer_metrics

7layers
8non_trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
9	variables
:regularization_losses
;trainable_variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "lambda_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_8", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4QAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
�

=axis
	&gamma
'beta
(moving_mean
)moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

*kernel
+bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
�


,kernel
-bias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
�
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_25", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
�


.kernel
/bias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
�
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
�
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
�
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
�	

0kernel
1bias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
�
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
�	

2kernel
3bias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
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
8
�0
�1
�2"
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
	variables
regularization_losses
trainable_variables
rmetrics
slayer_regularization_losses
tlayer_metrics

ulayers
vnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_26/kernel
:2dense_26/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
regularization_losses
wlayer_regularization_losses
xmetrics
trainable_variables
ylayer_metrics

zlayers
{non_trainable_variables
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
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
*:( 2conv2d_24/kernel
: 2conv2d_24/bias
+:) �2conv2d_25/kernel
:�2conv2d_25/bias
,:*��2conv2d_26/kernel
:�2conv2d_26/bias
$:"���2dense_24/kernel
:�2dense_24/bias
#:!
��2dense_25/kernel
:�2dense_25/bias
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
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
�
9	variables
:regularization_losses
~layer_regularization_losses
metrics
;trainable_variables
�layer_metrics
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
>	variables
?regularization_losses
 �layer_regularization_losses
�metrics
@trainable_variables
�layer_metrics
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
�
B	variables
Cregularization_losses
 �layer_regularization_losses
�metrics
Dtrainable_variables
�layer_metrics
�layers
�non_trainable_variables
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
F	variables
Gregularization_losses
 �layer_regularization_losses
�metrics
Htrainable_variables
�layer_metrics
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
�
J	variables
Kregularization_losses
 �layer_regularization_losses
�metrics
Ltrainable_variables
�layer_metrics
�layers
�non_trainable_variables
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
N	variables
Oregularization_losses
 �layer_regularization_losses
�metrics
Ptrainable_variables
�layer_metrics
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
�
R	variables
Sregularization_losses
 �layer_regularization_losses
�metrics
Ttrainable_variables
�layer_metrics
�layers
�non_trainable_variables
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
V	variables
Wregularization_losses
 �layer_regularization_losses
�metrics
Xtrainable_variables
�layer_metrics
�layers
�non_trainable_variables
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
Z	variables
[regularization_losses
 �layer_regularization_losses
�metrics
\trainable_variables
�layer_metrics
�layers
�non_trainable_variables
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
^	variables
_regularization_losses
 �layer_regularization_losses
�metrics
`trainable_variables
�layer_metrics
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
�
b	variables
cregularization_losses
 �layer_regularization_losses
�metrics
dtrainable_variables
�layer_metrics
�layers
�non_trainable_variables
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
f	variables
gregularization_losses
 �layer_regularization_losses
�metrics
htrainable_variables
�layer_metrics
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
�
j	variables
kregularization_losses
 �layer_regularization_losses
�metrics
ltrainable_variables
�layer_metrics
�layers
�non_trainable_variables
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
n	variables
oregularization_losses
 �layer_regularization_losses
�metrics
ptrainable_variables
�layer_metrics
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
(0
)1"
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
(0
)1"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
':%	�2Adam/dense_26/kernel/m
 :2Adam/dense_26/bias/m
.:,2"Adam/batch_normalization_8/gamma/m
-:+2!Adam/batch_normalization_8/beta/m
/:- 2Adam/conv2d_24/kernel/m
!: 2Adam/conv2d_24/bias/m
0:. �2Adam/conv2d_25/kernel/m
": �2Adam/conv2d_25/bias/m
1:/��2Adam/conv2d_26/kernel/m
": �2Adam/conv2d_26/bias/m
):'���2Adam/dense_24/kernel/m
!:�2Adam/dense_24/bias/m
(:&
��2Adam/dense_25/kernel/m
!:�2Adam/dense_25/bias/m
':%	�2Adam/dense_26/kernel/v
 :2Adam/dense_26/bias/v
.:,2"Adam/batch_normalization_8/gamma/v
-:+2!Adam/batch_normalization_8/beta/v
/:- 2Adam/conv2d_24/kernel/v
!: 2Adam/conv2d_24/bias/v
0:. �2Adam/conv2d_25/kernel/v
": �2Adam/conv2d_25/bias/v
1:/��2Adam/conv2d_26/kernel/v
": �2Adam/conv2d_26/bias/v
):'���2Adam/dense_24/kernel/v
!:�2Adam/dense_24/bias/v
(:&
��2Adam/dense_25/kernel/v
!:�2Adam/dense_25/bias/v
�2�
"__inference__wrapped_model_7364097�
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
�2�
@__inference_CNN_layer_call_and_return_conditional_losses_7365516
@__inference_CNN_layer_call_and_return_conditional_losses_7365627
@__inference_CNN_layer_call_and_return_conditional_losses_7365717
@__inference_CNN_layer_call_and_return_conditional_losses_7365828�
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
%__inference_CNN_layer_call_fn_7365865
%__inference_CNN_layer_call_fn_7365902
%__inference_CNN_layer_call_fn_7365939
%__inference_CNN_layer_call_fn_7365976�
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
__inference_call_7256974
__inference_call_7257046
__inference_call_7257118�
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366077
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366181
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366264
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366368�
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
.__inference_sequential_8_layer_call_fn_7366401
.__inference_sequential_8_layer_call_fn_7366434
.__inference_sequential_8_layer_call_fn_7366467
.__inference_sequential_8_layer_call_fn_7366500�
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
E__inference_dense_26_layer_call_and_return_conditional_losses_7366511�
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
*__inference_dense_26_layer_call_fn_7366520�
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
%__inference_signature_wrapper_7365426input_1"�
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
E__inference_lambda_8_layer_call_and_return_conditional_losses_7366528
E__inference_lambda_8_layer_call_and_return_conditional_losses_7366536�
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
*__inference_lambda_8_layer_call_fn_7366541
*__inference_lambda_8_layer_call_fn_7366546�
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366564
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366582
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366600
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366618�
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
7__inference_batch_normalization_8_layer_call_fn_7366631
7__inference_batch_normalization_8_layer_call_fn_7366644
7__inference_batch_normalization_8_layer_call_fn_7366657
7__inference_batch_normalization_8_layer_call_fn_7366670�
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
F__inference_conv2d_24_layer_call_and_return_conditional_losses_7366693�
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
+__inference_conv2d_24_layer_call_fn_7366702�
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
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_7364229�
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
2__inference_max_pooling2d_24_layer_call_fn_7364235�
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
F__inference_conv2d_25_layer_call_and_return_conditional_losses_7366713�
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
+__inference_conv2d_25_layer_call_fn_7366722�
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
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_7364241�
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
2__inference_max_pooling2d_25_layer_call_fn_7364247�
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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_7366733�
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
+__inference_conv2d_26_layer_call_fn_7366742�
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
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_7364253�
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
2__inference_max_pooling2d_26_layer_call_fn_7364259�
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
G__inference_dropout_24_layer_call_and_return_conditional_losses_7366747
G__inference_dropout_24_layer_call_and_return_conditional_losses_7366759�
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
,__inference_dropout_24_layer_call_fn_7366764
,__inference_dropout_24_layer_call_fn_7366769�
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
F__inference_flatten_8_layer_call_and_return_conditional_losses_7366775�
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
+__inference_flatten_8_layer_call_fn_7366780�
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
E__inference_dense_24_layer_call_and_return_conditional_losses_7366803�
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
*__inference_dense_24_layer_call_fn_7366812�
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
G__inference_dropout_25_layer_call_and_return_conditional_losses_7366817
G__inference_dropout_25_layer_call_and_return_conditional_losses_7366829�
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
,__inference_dropout_25_layer_call_fn_7366834
,__inference_dropout_25_layer_call_fn_7366839�
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
E__inference_dense_25_layer_call_and_return_conditional_losses_7366862�
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
*__inference_dense_25_layer_call_fn_7366871�
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
G__inference_dropout_26_layer_call_and_return_conditional_losses_7366876
G__inference_dropout_26_layer_call_and_return_conditional_losses_7366888�
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
,__inference_dropout_26_layer_call_fn_7366893
,__inference_dropout_26_layer_call_fn_7366898�
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
__inference_loss_fn_0_7366909�
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
__inference_loss_fn_1_7366920�
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
__inference_loss_fn_2_7366931�
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
@__inference_CNN_layer_call_and_return_conditional_losses_7365516v&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p 
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_7365627v&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_7365717w&'()*+,-./0123<�9
2�/
)�&
input_1���������KK
p 
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_7365828w&'()*+,-./0123<�9
2�/
)�&
input_1���������KK
p
� "%�"
�
0���������
� �
%__inference_CNN_layer_call_fn_7365865j&'()*+,-./0123<�9
2�/
)�&
input_1���������KK
p 
� "�����������
%__inference_CNN_layer_call_fn_7365902i&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p 
� "�����������
%__inference_CNN_layer_call_fn_7365939i&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p
� "�����������
%__inference_CNN_layer_call_fn_7365976j&'()*+,-./0123<�9
2�/
)�&
input_1���������KK
p
� "�����������
"__inference__wrapped_model_7364097�&'()*+,-./01238�5
.�+
)�&
input_1���������KK
� "3�0
.
output_1"�
output_1����������
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366564�&'()M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366582�&'()M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366600r&'();�8
1�.
(�%
inputs���������KK
p 
� "-�*
#� 
0���������KK
� �
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7366618r&'();�8
1�.
(�%
inputs���������KK
p
� "-�*
#� 
0���������KK
� �
7__inference_batch_normalization_8_layer_call_fn_7366631�&'()M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
7__inference_batch_normalization_8_layer_call_fn_7366644�&'()M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
7__inference_batch_normalization_8_layer_call_fn_7366657e&'();�8
1�.
(�%
inputs���������KK
p 
� " ����������KK�
7__inference_batch_normalization_8_layer_call_fn_7366670e&'();�8
1�.
(�%
inputs���������KK
p
� " ����������KKu
__inference_call_7256974Y&'()*+,-./01233�0
)�&
 �
inputs�KK
p
� "�	�u
__inference_call_7257046Y&'()*+,-./01233�0
)�&
 �
inputs�KK
p 
� "�	��
__inference_call_7257118i&'()*+,-./0123;�8
1�.
(�%
inputs���������KK
p 
� "�����������
F__inference_conv2d_24_layer_call_and_return_conditional_losses_7366693l*+7�4
-�*
(�%
inputs���������KK
� "-�*
#� 
0���������KK 
� �
+__inference_conv2d_24_layer_call_fn_7366702_*+7�4
-�*
(�%
inputs���������KK
� " ����������KK �
F__inference_conv2d_25_layer_call_and_return_conditional_losses_7366713m,-7�4
-�*
(�%
inputs���������%% 
� ".�+
$�!
0���������%%�
� �
+__inference_conv2d_25_layer_call_fn_7366722`,-7�4
-�*
(�%
inputs���������%% 
� "!����������%%��
F__inference_conv2d_26_layer_call_and_return_conditional_losses_7366733n./8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_conv2d_26_layer_call_fn_7366742a./8�5
.�+
)�&
inputs����������
� "!������������
E__inference_dense_24_layer_call_and_return_conditional_losses_7366803_011�.
'�$
"�
inputs�����������
� "&�#
�
0����������
� �
*__inference_dense_24_layer_call_fn_7366812R011�.
'�$
"�
inputs�����������
� "������������
E__inference_dense_25_layer_call_and_return_conditional_losses_7366862^230�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_25_layer_call_fn_7366871Q230�-
&�#
!�
inputs����������
� "������������
E__inference_dense_26_layer_call_and_return_conditional_losses_7366511]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_26_layer_call_fn_7366520P0�-
&�#
!�
inputs����������
� "�����������
G__inference_dropout_24_layer_call_and_return_conditional_losses_7366747n<�9
2�/
)�&
inputs���������		�
p 
� ".�+
$�!
0���������		�
� �
G__inference_dropout_24_layer_call_and_return_conditional_losses_7366759n<�9
2�/
)�&
inputs���������		�
p
� ".�+
$�!
0���������		�
� �
,__inference_dropout_24_layer_call_fn_7366764a<�9
2�/
)�&
inputs���������		�
p 
� "!����������		��
,__inference_dropout_24_layer_call_fn_7366769a<�9
2�/
)�&
inputs���������		�
p
� "!����������		��
G__inference_dropout_25_layer_call_and_return_conditional_losses_7366817^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_25_layer_call_and_return_conditional_losses_7366829^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_25_layer_call_fn_7366834Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_25_layer_call_fn_7366839Q4�1
*�'
!�
inputs����������
p
� "������������
G__inference_dropout_26_layer_call_and_return_conditional_losses_7366876^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_26_layer_call_and_return_conditional_losses_7366888^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_26_layer_call_fn_7366893Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_26_layer_call_fn_7366898Q4�1
*�'
!�
inputs����������
p
� "������������
F__inference_flatten_8_layer_call_and_return_conditional_losses_7366775c8�5
.�+
)�&
inputs���������		�
� "'�$
�
0�����������
� �
+__inference_flatten_8_layer_call_fn_7366780V8�5
.�+
)�&
inputs���������		�
� "�������������
E__inference_lambda_8_layer_call_and_return_conditional_losses_7366528p?�<
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
E__inference_lambda_8_layer_call_and_return_conditional_losses_7366536p?�<
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
*__inference_lambda_8_layer_call_fn_7366541c?�<
5�2
(�%
inputs���������KK

 
p 
� " ����������KK�
*__inference_lambda_8_layer_call_fn_7366546c?�<
5�2
(�%
inputs���������KK

 
p
� " ����������KK<
__inference_loss_fn_0_7366909*�

� 
� "� <
__inference_loss_fn_1_73669200�

� 
� "� <
__inference_loss_fn_2_73669312�

� 
� "� �
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_7364229�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_24_layer_call_fn_7364235�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_7364241�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_25_layer_call_fn_7364247�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_7364253�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_26_layer_call_fn_7364259�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366077y&'()*+,-./0123?�<
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366181y&'()*+,-./0123?�<
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366264�&'()*+,-./0123G�D
=�:
0�-
lambda_8_input���������KK
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_8_layer_call_and_return_conditional_losses_7366368�&'()*+,-./0123G�D
=�:
0�-
lambda_8_input���������KK
p

 
� "&�#
�
0����������
� �
.__inference_sequential_8_layer_call_fn_7366401t&'()*+,-./0123G�D
=�:
0�-
lambda_8_input���������KK
p 

 
� "������������
.__inference_sequential_8_layer_call_fn_7366434l&'()*+,-./0123?�<
5�2
(�%
inputs���������KK
p 

 
� "������������
.__inference_sequential_8_layer_call_fn_7366467l&'()*+,-./0123?�<
5�2
(�%
inputs���������KK
p

 
� "������������
.__inference_sequential_8_layer_call_fn_7366500t&'()*+,-./0123G�D
=�:
0�-
lambda_8_input���������KK
p

 
� "������������
%__inference_signature_wrapper_7365426�&'()*+,-./0123C�@
� 
9�6
4
input_1)�&
input_1���������KK"3�0
.
output_1"�
output_1���������