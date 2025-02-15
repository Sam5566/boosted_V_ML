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
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	�*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
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
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
�
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
: *
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
: *
dtype0
�
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*!
shared_nameconv2d_10/kernel
~
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*'
_output_shapes
: �*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:�*
dtype0
�
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:�*
dtype0
{
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*
shared_namedense_9/kernel
t
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*!
_output_shapes
:���*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:�*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
��*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:�*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:*
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
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_11/kernel/m
�
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/m
�
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:*
dtype0
�
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/m
�
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_9/kernel/m
�
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_9/bias/m
y
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_10/kernel/m
�
+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_10/bias/m
|
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_11/kernel/m
�
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_11/bias/m
|
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*&
shared_nameAdam/dense_9/kernel/m
�
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*!
_output_shapes
:���*
dtype0

Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_9/bias/m
x
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_10/kernel/m
�
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_10/bias/m
z
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_11/kernel/v
�
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/v
�
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:*
dtype0
�
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/v
�
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_9/kernel/v
�
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_9/bias/v
y
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_10/kernel/v
�
+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_10/bias/v
|
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_11/kernel/v
�
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_11/bias/v
|
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*&
shared_nameAdam/dense_9/kernel/v
�
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*!
_output_shapes
:���*
dtype0

Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_9/bias/v
x
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_10/kernel/v
�
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_10/bias/v
z
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�`
value�_B�_ B�_
�

h2ptjl
_output
	optimizer
regularization_losses
trainable_variables
	variables
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
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem�m�&m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�v�v�&v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�
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
12
13
v
&0
'1
22
33
(4
)5
*6
+7
,8
-9
.10
/11
012
113
14
15
�
regularization_losses
4metrics
trainable_variables
5layer_regularization_losses

6layers
7layer_metrics
8non_trainable_variables
	variables
 
R
9regularization_losses
:trainable_variables
;	variables
<	keras_api
�
=axis
	&gamma
'beta
2moving_mean
3moving_variance
>regularization_losses
?trainable_variables
@	variables
A	keras_api
h

(kernel
)bias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
R
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
h

*kernel
+bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
R
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
h

,kernel
-bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
R
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
R
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
R
^regularization_losses
_trainable_variables
`	variables
a	keras_api
h

.kernel
/bias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
R
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
h

0kernel
1bias
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
R
nregularization_losses
otrainable_variables
p	variables
q	keras_api
 
V
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
f
&0
'1
22
33
(4
)5
*6
+7
,8
-9
.10
/11
012
113
�
regularization_losses
rmetrics
trainable_variables
slayer_regularization_losses

tlayers
ulayer_metrics
vnon_trainable_variables
	variables
NL
VARIABLE_VALUEdense_11/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_11/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
wmetrics
trainable_variables
xlayer_regularization_losses

ylayers
zlayer_metrics
{non_trainable_variables
	variables
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
a_
VARIABLE_VALUEbatch_normalization_3/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_3/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_9/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_9/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_10/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_10/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_11/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_11/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_9/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_9/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_10/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_10/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_3/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_3/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE

|0
}1
 

0
1
 

20
31
 
 
 
�
9regularization_losses
~metrics
:trainable_variables
layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
;	variables
 
 

&0
'1

&0
'1
22
33
�
>regularization_losses
�metrics
?trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
@	variables
 

(0
)1

(0
)1
�
Bregularization_losses
�metrics
Ctrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
D	variables
 
 
 
�
Fregularization_losses
�metrics
Gtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
H	variables
 

*0
+1

*0
+1
�
Jregularization_losses
�metrics
Ktrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
L	variables
 
 
 
�
Nregularization_losses
�metrics
Otrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
P	variables
 

,0
-1

,0
-1
�
Rregularization_losses
�metrics
Strainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
T	variables
 
 
 
�
Vregularization_losses
�metrics
Wtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
X	variables
 
 
 
�
Zregularization_losses
�metrics
[trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
\	variables
 
 
 
�
^regularization_losses
�metrics
_trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
`	variables
 

.0
/1

.0
/1
�
bregularization_losses
�metrics
ctrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
d	variables
 
 
 
�
fregularization_losses
�metrics
gtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
h	variables
 

00
11

00
11
�
jregularization_losses
�metrics
ktrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
l	variables
 
 
 
�
nregularization_losses
�metrics
otrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
p	variables
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

20
31
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
20
31
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
VARIABLE_VALUEAdam/dense_11/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_11/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_9/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_9/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_10/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_10/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_11/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_11/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_9/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_9/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_10/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_10/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_11/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_11/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_9/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_9/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_10/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_10/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_11/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_11/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_9/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_9/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_10/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_10/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������KK*
dtype0*$
shape:���������KK
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*
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
GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_707723
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOpConst*B
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
GPU2 *0J 8� *(
f#R!
__inference__traced_save_709410
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_3/gammabatch_normalization_3/betaconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancetotalcounttotal_1count_1Adam/dense_11/kernel/mAdam/dense_11/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/vAdam/dense_11/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/v*A
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
GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_709579�
�
�
-__inference_sequential_3_layer_call_fn_708764

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
GPU2 *0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_7070722
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
L
0__inference_max_pooling2d_9_layer_call_fn_706532

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
GPU2 *0J 8� *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_7065262
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
�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_706635

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
�`
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_707072

inputs*
batch_normalization_3_707012:*
batch_normalization_3_707014:*
batch_normalization_3_707016:*
batch_normalization_3_707018:)
conv2d_9_707021: 
conv2d_9_707023: +
conv2d_10_707027: �
conv2d_10_707029:	�,
conv2d_11_707033:��
conv2d_11_707035:	�#
dense_9_707041:���
dense_9_707043:	�#
dense_10_707047:
��
dense_10_707049:	�
identity��-batch_normalization_3/StatefulPartitionedCall�!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp� dense_10/StatefulPartitionedCall�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_9/StatefulPartitionedCall�0dense_9/kernel/Regularizer/Square/ReadVariableOp�"dropout_10/StatefulPartitionedCall�"dropout_11/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8� *M
fHRF
D__inference_lambda_3_layer_call_and_return_conditional_losses_7069702
lambda_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0batch_normalization_3_707012batch_normalization_3_707014batch_normalization_3_707016batch_normalization_3_707018*
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
GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7069432/
-batch_normalization_3/StatefulPartitionedCall�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_9_707021conv2d_9_707023*
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
GPU2 *0J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_7066172"
 conv2d_9/StatefulPartitionedCall�
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_7065262!
max_pooling2d_9/PartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0conv2d_10_707027conv2d_10_707029*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_7066352#
!conv2d_10/StatefulPartitionedCall�
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_7065382"
 max_pooling2d_10/PartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_707033conv2d_11_707035*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_7066532#
!conv2d_11/StatefulPartitionedCall�
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_7065502"
 max_pooling2d_11/PartitionedCall�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
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
GPU2 *0J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_7068772#
!dropout_9/StatefulPartitionedCall�
flatten_3/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_7066732
flatten_3/PartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_707041dense_9_707043*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_7066922!
dense_9/StatefulPartitionedCall�
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
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
GPU2 *0J 8� *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_7068382$
"dropout_10/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_10_707047dense_10_707049*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_7067222"
 dense_10/StatefulPartitionedCall�
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
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
GPU2 *0J 8� *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_7068052$
"dropout_11/StatefulPartitionedCall�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_707021*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_707041*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_707047* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentity+dropout_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_9/StatefulPartitionedCall1^dense_9/kernel/Regularizer/Square/ReadVariableOp#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�j
�
__inference__traced_save_709410
file_prefix.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*�
value�B�6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�: :	�:: : : : : ::: : : �:�:��:�:���:�:
��:�::: : : : :	�:::: : : �:�:��:�:���:�:
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
::,
(
&
_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: �:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:'#
!
_output_shapes
:���:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�: 

_output_shapes
:: 

_output_shapes
::
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
�
G
+__inference_dropout_10_layer_call_fn_709131

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
GPU2 *0J 8� *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_7067032
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
$__inference_signature_wrapper_707723
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
GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_7063942
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
�
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_706703

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
�u
�
__inference_call_591983

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: �E
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	�Q
5sequential_3_conv2d_11_conv2d_readvariableop_resource:��E
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	�H
3sequential_3_dense_9_matmul_readvariableop_resource:���C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�H
4sequential_3_dense_10_matmul_readvariableop_resource:
��D
5sequential_3_dense_10_biasadd_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_3/ReadVariableOp�3sequential_3/batch_normalization_3/ReadVariableOp_1�-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�,sequential_3/conv2d_10/Conv2D/ReadVariableOp�-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�,sequential_3/conv2d_11/Conv2D/ReadVariableOp�,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�+sequential_3/conv2d_9/Conv2D/ReadVariableOp�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp�
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack�
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1�
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2�
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice�
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOp�
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3�
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOp�
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D�
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/BiasAdd�
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/Relu�
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool�
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOp�
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D�
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_3/conv2d_10/BiasAdd�
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_3/conv2d_10/Relu�
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool�
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOp�
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D�
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_3/conv2d_11/BiasAdd�
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_3/conv2d_11/Relu�
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool�
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2!
sequential_3/dropout_9/Identity�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/MatMul�
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/BiasAdd�
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/Relu�
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_3/dropout_10/Identity�
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/MatMul�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/BiasAdd�
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/Relu�
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_3/dropout_11/Identity�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_11/Softmax�
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_3/batch_normalization_3/ReadVariableOp1sequential_3/batch_normalization_3/ReadVariableOp2j
3sequential_3/batch_normalization_3/ReadVariableOp_13sequential_3/batch_normalization_3/ReadVariableOp_12^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�s
�
__inference_call_591839

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: �E
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	�Q
5sequential_3_conv2d_11_conv2d_readvariableop_resource:��E
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	�H
3sequential_3_dense_9_matmul_readvariableop_resource:���C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�H
4sequential_3_dense_10_matmul_readvariableop_resource:
��D
5sequential_3_dense_10_biasadd_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_3/ReadVariableOp�3sequential_3/batch_normalization_3/ReadVariableOp_1�-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�,sequential_3/conv2d_10/Conv2D/ReadVariableOp�-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�,sequential_3/conv2d_11/Conv2D/ReadVariableOp�,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�+sequential_3/conv2d_9/Conv2D/ReadVariableOp�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp�
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack�
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1�
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2�
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice�
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOp�
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3�
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOp�
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D�
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2
sequential_3/conv2d_9/BiasAdd�
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_3/conv2d_9/Relu�
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool�
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOp�
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D�
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_3/conv2d_10/BiasAdd�
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_3/conv2d_10/Relu�
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool�
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOp�
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D�
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_3/conv2d_11/BiasAdd�
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_3/conv2d_11/Relu�
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool�
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*(
_output_shapes
:�		�2!
sequential_3/dropout_9/Identity�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*!
_output_shapes
:���2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_9/MatMul�
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_9/BiasAdd�
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_9/Relu�
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_3/dropout_10/Identity�
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_10/MatMul�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_10/BiasAdd�
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_10/Relu�
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_3/dropout_11/Identity�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_11/BiasAddt
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_11/Softmax�
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:�KK: : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_3/batch_normalization_3/ReadVariableOp1sequential_3/batch_normalization_3/ReadVariableOp2j
3sequential_3/batch_normalization_3/ReadVariableOp_13sequential_3/batch_normalization_3/ReadVariableOp_12^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_706617

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
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
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
)__inference_dense_10_layer_call_fn_709168

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
GPU2 *0J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_7067222
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
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708915

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
�u
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_708374

inputs;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: C
(conv2d_10_conv2d_readvariableop_resource: �8
)conv2d_10_biasadd_readvariableop_resource:	�D
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�;
&dense_9_matmul_readvariableop_resource:���6
'dense_9_biasadd_readvariableop_resource:	�;
'dense_10_matmul_readvariableop_resource:
��7
(dense_10_biasadd_readvariableop_resource:	�
identity��5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�
lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_3/strided_slice/stack�
lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_3/strided_slice/stack_1�
lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_3/strided_slice/stack_2�
lambda_3/strided_sliceStridedSliceinputs%lambda_3/strided_slice/stack:output:0'lambda_3/strided_slice/stack_1:output:0'lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_3/strided_slice�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3lambda_3/strided_slice:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_9/Conv2D�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_9/Relu�
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2D max_pooling2d_9/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_10/Relu�
max_pooling2d_10/MaxPoolMaxPoolconv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_11/Conv2D/ReadVariableOp�
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_11/Conv2D�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_11/Relu�
max_pooling2d_11/MaxPoolMaxPoolconv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPool�
dropout_9/IdentityIdentity!max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2
dropout_9/Identitys
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_3/Const�
flatten_3/ReshapeReshapedropout_9/Identity:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_3/Reshape�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9/Relu�
dropout_10/IdentityIdentitydense_9/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_10/Identity�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldropout_10/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_10/Relu�
dropout_11/IdentityIdentitydense_10/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_11/Identity�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentitydropout_11/Identity:output:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�

�
D__inference_dense_11_layer_call_and_return_conditional_losses_708808

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
D__inference_dense_11_layer_call_and_return_conditional_losses_707311

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
__inference_loss_fn_1_709217N
9dense_9_kernel_regularizer_square_readvariableop_resource:���
identity��0dense_9/kernel/Regularizer/Square/ReadVariableOp�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_9_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
IdentityIdentity"dense_9/kernel/Regularizer/mul:z:01^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp
�
F
*__inference_dropout_9_layer_call_fn_709061

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
GPU2 *0J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_7066652
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
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708897

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
�
�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_708990

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
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
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
C__inference_dense_9_layer_call_and_return_conditional_losses_706692

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�
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
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_709126

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
�
__inference_loss_fn_0_709206T
:conv2d_9_kernel_regularizer_square_readvariableop_resource: 
identity��1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_9_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
IdentityIdentity#conv2d_9/kernel/Regularizer/mul:z:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp
�
h
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_706538

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
d
+__inference_dropout_11_layer_call_fn_709195

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
GPU2 *0J 8� *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_7068052
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
�
$__inference_CNN_layer_call_fn_708236

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
GPU2 *0J 8� *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_7074762
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
��
�
?__inference_CNN_layer_call_and_return_conditional_losses_708125
input_1H
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: �E
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	�Q
5sequential_3_conv2d_11_conv2d_readvariableop_resource:��E
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	�H
3sequential_3_dense_9_matmul_readvariableop_resource:���C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�H
4sequential_3_dense_10_matmul_readvariableop_resource:
��D
5sequential_3_dense_10_biasadd_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�1sequential_3/batch_normalization_3/AssignNewValue�3sequential_3/batch_normalization_3/AssignNewValue_1�Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_3/ReadVariableOp�3sequential_3/batch_normalization_3/ReadVariableOp_1�-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�,sequential_3/conv2d_10/Conv2D/ReadVariableOp�-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�,sequential_3/conv2d_11/Conv2D/ReadVariableOp�,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�+sequential_3/conv2d_9/Conv2D/ReadVariableOp�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp�
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack�
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1�
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2�
#sequential_3/lambda_3/strided_sliceStridedSliceinput_12sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice�
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOp�
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<25
3sequential_3/batch_normalization_3/FusedBatchNormV3�
1sequential_3/batch_normalization_3/AssignNewValueAssignVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource@sequential_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_3/batch_normalization_3/AssignNewValue�
3sequential_3/batch_normalization_3/AssignNewValue_1AssignVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceDsequential_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0E^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_3/batch_normalization_3/AssignNewValue_1�
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOp�
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D�
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/BiasAdd�
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/Relu�
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool�
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOp�
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D�
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_3/conv2d_10/BiasAdd�
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_3/conv2d_10/Relu�
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool�
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOp�
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D�
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_3/conv2d_11/BiasAdd�
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_3/conv2d_11/Relu�
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool�
$sequential_3/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2&
$sequential_3/dropout_9/dropout/Const�
"sequential_3/dropout_9/dropout/MulMul.sequential_3/max_pooling2d_11/MaxPool:output:0-sequential_3/dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2$
"sequential_3/dropout_9/dropout/Mul�
$sequential_3/dropout_9/dropout/ShapeShape.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_9/dropout/Shape�
;sequential_3/dropout_9/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype02=
;sequential_3/dropout_9/dropout/random_uniform/RandomUniform�
-sequential_3/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-sequential_3/dropout_9/dropout/GreaterEqual/y�
+sequential_3/dropout_9/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_9/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2-
+sequential_3/dropout_9/dropout/GreaterEqual�
#sequential_3/dropout_9/dropout/CastCast/sequential_3/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2%
#sequential_3/dropout_9/dropout/Cast�
$sequential_3/dropout_9/dropout/Mul_1Mul&sequential_3/dropout_9/dropout/Mul:z:0'sequential_3/dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2&
$sequential_3/dropout_9/dropout/Mul_1�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/dropout/Mul_1:z:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/MatMul�
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/BiasAdd�
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/Relu�
%sequential_3/dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_10/dropout/Const�
#sequential_3/dropout_10/dropout/MulMul'sequential_3/dense_9/Relu:activations:0.sequential_3/dropout_10/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_3/dropout_10/dropout/Mul�
%sequential_3/dropout_10/dropout/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_10/dropout/Shape�
<sequential_3/dropout_10/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_10/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_3/dropout_10/dropout/random_uniform/RandomUniform�
.sequential_3/dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_10/dropout/GreaterEqual/y�
,sequential_3/dropout_10/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_10/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_3/dropout_10/dropout/GreaterEqual�
$sequential_3/dropout_10/dropout/CastCast0sequential_3/dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_3/dropout_10/dropout/Cast�
%sequential_3/dropout_10/dropout/Mul_1Mul'sequential_3/dropout_10/dropout/Mul:z:0(sequential_3/dropout_10/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_3/dropout_10/dropout/Mul_1�
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/dropout/Mul_1:z:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/MatMul�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/BiasAdd�
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/Relu�
%sequential_3/dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_11/dropout/Const�
#sequential_3/dropout_11/dropout/MulMul(sequential_3/dense_10/Relu:activations:0.sequential_3/dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_3/dropout_11/dropout/Mul�
%sequential_3/dropout_11/dropout/ShapeShape(sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_11/dropout/Shape�
<sequential_3/dropout_11/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_3/dropout_11/dropout/random_uniform/RandomUniform�
.sequential_3/dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_11/dropout/GreaterEqual/y�
,sequential_3/dropout_11/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_11/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_3/dropout_11/dropout/GreaterEqual�
$sequential_3/dropout_11/dropout/CastCast0sequential_3/dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_3/dropout_11/dropout/Cast�
%sequential_3/dropout_11/dropout/Mul_1Mul'sequential_3/dropout_11/dropout/Mul:z:0(sequential_3/dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_3/dropout_11/dropout/Mul_1�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)sequential_3/dropout_11/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_11/Softmax�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentitydense_11/Softmax:softmax:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp2^sequential_3/batch_normalization_3/AssignNewValue4^sequential_3/batch_normalization_3/AssignNewValue_1C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2f
1sequential_3/batch_normalization_3/AssignNewValue1sequential_3/batch_normalization_3/AssignNewValue2j
3sequential_3/batch_normalization_3/AssignNewValue_13sequential_3/batch_normalization_3/AssignNewValue_12�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_3/batch_normalization_3/ReadVariableOp1sequential_3/batch_normalization_3/ReadVariableOp2j
3sequential_3/batch_normalization_3/ReadVariableOp_13sequential_3/batch_normalization_3/ReadVariableOp_12^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_709030

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
��
�
?__inference_CNN_layer_call_and_return_conditional_losses_708014
input_1H
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: �E
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	�Q
5sequential_3_conv2d_11_conv2d_readvariableop_resource:��E
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	�H
3sequential_3_dense_9_matmul_readvariableop_resource:���C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�H
4sequential_3_dense_10_matmul_readvariableop_resource:
��D
5sequential_3_dense_10_biasadd_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_3/ReadVariableOp�3sequential_3/batch_normalization_3/ReadVariableOp_1�-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�,sequential_3/conv2d_10/Conv2D/ReadVariableOp�-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�,sequential_3/conv2d_11/Conv2D/ReadVariableOp�,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�+sequential_3/conv2d_9/Conv2D/ReadVariableOp�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp�
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack�
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1�
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2�
#sequential_3/lambda_3/strided_sliceStridedSliceinput_12sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice�
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOp�
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3�
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOp�
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D�
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/BiasAdd�
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/Relu�
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool�
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOp�
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D�
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_3/conv2d_10/BiasAdd�
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_3/conv2d_10/Relu�
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool�
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOp�
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D�
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_3/conv2d_11/BiasAdd�
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_3/conv2d_11/Relu�
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool�
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2!
sequential_3/dropout_9/Identity�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/MatMul�
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/BiasAdd�
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/Relu�
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_3/dropout_10/Identity�
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/MatMul�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/BiasAdd�
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/Relu�
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_3/dropout_11/Identity�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_11/Softmax�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentitydense_11/Softmax:softmax:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_3/batch_normalization_3/ReadVariableOp1sequential_3/batch_normalization_3/ReadVariableOp2j
3sequential_3/batch_normalization_3/ReadVariableOp_13sequential_3/batch_normalization_3/ReadVariableOp_12^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
$__inference_CNN_layer_call_fn_708199

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
GPU2 *0J 8� *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_7073362
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
�
�
D__inference_dense_10_layer_call_and_return_conditional_losses_706722

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_CNN_layer_call_fn_708273
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
GPU2 *0J 8� *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_7074762
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
�
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_706838

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
)__inference_dense_11_layer_call_fn_708817

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
GPU2 *0J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_7073112
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
E
)__inference_lambda_3_layer_call_fn_708843

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
GPU2 *0J 8� *M
fHRF
D__inference_lambda_3_layer_call_and_return_conditional_losses_7069702
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
�
E
)__inference_lambda_3_layer_call_fn_708838

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
GPU2 *0J 8� *M
fHRF
D__inference_lambda_3_layer_call_and_return_conditional_losses_7065712
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
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_709173

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
�
g
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_706526

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
`
D__inference_lambda_3_layer_call_and_return_conditional_losses_708825

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
`
D__inference_lambda_3_layer_call_and_return_conditional_losses_706571

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
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_706877

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
�
d
+__inference_dropout_10_layer_call_fn_709136

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
GPU2 *0J 8� *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_7068382
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
�u
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_708561
lambda_3_input;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: C
(conv2d_10_conv2d_readvariableop_resource: �8
)conv2d_10_biasadd_readvariableop_resource:	�D
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�;
&dense_9_matmul_readvariableop_resource:���6
'dense_9_biasadd_readvariableop_resource:	�;
'dense_10_matmul_readvariableop_resource:
��7
(dense_10_biasadd_readvariableop_resource:	�
identity��5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�
lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_3/strided_slice/stack�
lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_3/strided_slice/stack_1�
lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_3/strided_slice/stack_2�
lambda_3/strided_sliceStridedSlicelambda_3_input%lambda_3/strided_slice/stack:output:0'lambda_3/strided_slice/stack_1:output:0'lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_3/strided_slice�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3lambda_3/strided_slice:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_9/Conv2D�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_9/Relu�
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2D max_pooling2d_9/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_10/Relu�
max_pooling2d_10/MaxPoolMaxPoolconv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_11/Conv2D/ReadVariableOp�
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_11/Conv2D�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_11/Relu�
max_pooling2d_11/MaxPoolMaxPoolconv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPool�
dropout_9/IdentityIdentity!max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2
dropout_9/Identitys
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_3/Const�
flatten_3/ReshapeReshapedropout_9/Identity:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_3/Reshape�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9/Relu�
dropout_10/IdentityIdentitydense_9/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_10/Identity�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldropout_10/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_10/Relu�
dropout_11/IdentityIdentitydense_10/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_11/Identity�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentitydropout_11/Identity:output:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_3_input
��
�
?__inference_CNN_layer_call_and_return_conditional_losses_707924

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: �E
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	�Q
5sequential_3_conv2d_11_conv2d_readvariableop_resource:��E
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	�H
3sequential_3_dense_9_matmul_readvariableop_resource:���C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�H
4sequential_3_dense_10_matmul_readvariableop_resource:
��D
5sequential_3_dense_10_biasadd_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�1sequential_3/batch_normalization_3/AssignNewValue�3sequential_3/batch_normalization_3/AssignNewValue_1�Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_3/ReadVariableOp�3sequential_3/batch_normalization_3/ReadVariableOp_1�-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�,sequential_3/conv2d_10/Conv2D/ReadVariableOp�-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�,sequential_3/conv2d_11/Conv2D/ReadVariableOp�,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�+sequential_3/conv2d_9/Conv2D/ReadVariableOp�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp�
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack�
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1�
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2�
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice�
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOp�
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<25
3sequential_3/batch_normalization_3/FusedBatchNormV3�
1sequential_3/batch_normalization_3/AssignNewValueAssignVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource@sequential_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_3/batch_normalization_3/AssignNewValue�
3sequential_3/batch_normalization_3/AssignNewValue_1AssignVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceDsequential_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0E^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_3/batch_normalization_3/AssignNewValue_1�
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOp�
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D�
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/BiasAdd�
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/Relu�
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool�
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOp�
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D�
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_3/conv2d_10/BiasAdd�
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_3/conv2d_10/Relu�
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool�
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOp�
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D�
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_3/conv2d_11/BiasAdd�
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_3/conv2d_11/Relu�
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool�
$sequential_3/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2&
$sequential_3/dropout_9/dropout/Const�
"sequential_3/dropout_9/dropout/MulMul.sequential_3/max_pooling2d_11/MaxPool:output:0-sequential_3/dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2$
"sequential_3/dropout_9/dropout/Mul�
$sequential_3/dropout_9/dropout/ShapeShape.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_9/dropout/Shape�
;sequential_3/dropout_9/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype02=
;sequential_3/dropout_9/dropout/random_uniform/RandomUniform�
-sequential_3/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-sequential_3/dropout_9/dropout/GreaterEqual/y�
+sequential_3/dropout_9/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_9/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2-
+sequential_3/dropout_9/dropout/GreaterEqual�
#sequential_3/dropout_9/dropout/CastCast/sequential_3/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2%
#sequential_3/dropout_9/dropout/Cast�
$sequential_3/dropout_9/dropout/Mul_1Mul&sequential_3/dropout_9/dropout/Mul:z:0'sequential_3/dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2&
$sequential_3/dropout_9/dropout/Mul_1�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/dropout/Mul_1:z:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/MatMul�
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/BiasAdd�
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/Relu�
%sequential_3/dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_10/dropout/Const�
#sequential_3/dropout_10/dropout/MulMul'sequential_3/dense_9/Relu:activations:0.sequential_3/dropout_10/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_3/dropout_10/dropout/Mul�
%sequential_3/dropout_10/dropout/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_10/dropout/Shape�
<sequential_3/dropout_10/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_10/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_3/dropout_10/dropout/random_uniform/RandomUniform�
.sequential_3/dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_10/dropout/GreaterEqual/y�
,sequential_3/dropout_10/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_10/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_3/dropout_10/dropout/GreaterEqual�
$sequential_3/dropout_10/dropout/CastCast0sequential_3/dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_3/dropout_10/dropout/Cast�
%sequential_3/dropout_10/dropout/Mul_1Mul'sequential_3/dropout_10/dropout/Mul:z:0(sequential_3/dropout_10/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_3/dropout_10/dropout/Mul_1�
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/dropout/Mul_1:z:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/MatMul�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/BiasAdd�
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/Relu�
%sequential_3/dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_11/dropout/Const�
#sequential_3/dropout_11/dropout/MulMul(sequential_3/dense_10/Relu:activations:0.sequential_3/dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_3/dropout_11/dropout/Mul�
%sequential_3/dropout_11/dropout/ShapeShape(sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_11/dropout/Shape�
<sequential_3/dropout_11/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_3/dropout_11/dropout/random_uniform/RandomUniform�
.sequential_3/dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_11/dropout/GreaterEqual/y�
,sequential_3/dropout_11/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_11/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_3/dropout_11/dropout/GreaterEqual�
$sequential_3/dropout_11/dropout/CastCast0sequential_3/dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_3/dropout_11/dropout/Cast�
%sequential_3/dropout_11/dropout/Mul_1Mul'sequential_3/dropout_11/dropout/Mul:z:0(sequential_3/dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_3/dropout_11/dropout/Mul_1�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)sequential_3/dropout_11/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_11/Softmax�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentitydense_11/Softmax:softmax:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp2^sequential_3/batch_normalization_3/AssignNewValue4^sequential_3/batch_normalization_3/AssignNewValue_1C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2f
1sequential_3/batch_normalization_3/AssignNewValue1sequential_3/batch_normalization_3/AssignNewValue2j
3sequential_3/batch_normalization_3/AssignNewValue_13sequential_3/batch_normalization_3/AssignNewValue_12�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_3/batch_normalization_3/ReadVariableOp1sequential_3/batch_normalization_3/ReadVariableOp2j
3sequential_3/batch_normalization_3/ReadVariableOp_13sequential_3/batch_normalization_3/ReadVariableOp_12^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�u
�
__inference_call_589853

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: �E
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	�Q
5sequential_3_conv2d_11_conv2d_readvariableop_resource:��E
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	�H
3sequential_3_dense_9_matmul_readvariableop_resource:���C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�H
4sequential_3_dense_10_matmul_readvariableop_resource:
��D
5sequential_3_dense_10_biasadd_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_3/ReadVariableOp�3sequential_3/batch_normalization_3/ReadVariableOp_1�-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�,sequential_3/conv2d_10/Conv2D/ReadVariableOp�-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�,sequential_3/conv2d_11/Conv2D/ReadVariableOp�,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�+sequential_3/conv2d_9/Conv2D/ReadVariableOp�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp�
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack�
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1�
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2�
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice�
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOp�
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3�
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOp�
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D�
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/BiasAdd�
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/Relu�
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool�
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOp�
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D�
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_3/conv2d_10/BiasAdd�
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_3/conv2d_10/Relu�
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool�
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOp�
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D�
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_3/conv2d_11/BiasAdd�
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_3/conv2d_11/Relu�
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool�
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2!
sequential_3/dropout_9/Identity�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/MatMul�
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/BiasAdd�
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/Relu�
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_3/dropout_10/Identity�
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/MatMul�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/BiasAdd�
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/Relu�
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_3/dropout_11/Identity�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_11/Softmax�
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_3/batch_normalization_3/ReadVariableOp1sequential_3/batch_normalization_3/ReadVariableOp2j
3sequential_3/batch_normalization_3/ReadVariableOp_13sequential_3/batch_normalization_3/ReadVariableOp_12^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_706653

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
ә
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_708665
lambda_3_input;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: C
(conv2d_10_conv2d_readvariableop_resource: �8
)conv2d_10_biasadd_readvariableop_resource:	�D
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�;
&dense_9_matmul_readvariableop_resource:���6
'dense_9_biasadd_readvariableop_resource:	�;
'dense_10_matmul_readvariableop_resource:
��7
(dense_10_biasadd_readvariableop_resource:	�
identity��$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�
lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_3/strided_slice/stack�
lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_3/strided_slice/stack_1�
lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_3/strided_slice/stack_2�
lambda_3/strided_sliceStridedSlicelambda_3_input%lambda_3/strided_slice/stack:output:0'lambda_3/strided_slice/stack_1:output:0'lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_3/strided_slice�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3lambda_3/strided_slice:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_3/FusedBatchNormV3�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_9/Conv2D�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_9/Relu�
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2D max_pooling2d_9/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_10/Relu�
max_pooling2d_10/MaxPoolMaxPoolconv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_11/Conv2D/ReadVariableOp�
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_11/Conv2D�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_11/Relu�
max_pooling2d_11/MaxPoolMaxPoolconv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPoolw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_9/dropout/Const�
dropout_9/dropout/MulMul!max_pooling2d_11/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2
dropout_9/dropout/Mul�
dropout_9/dropout/ShapeShape!max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform�
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2"
 dropout_9/dropout/GreaterEqual/y�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2 
dropout_9/dropout/GreaterEqual�
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout_9/dropout/Cast�
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout_9/dropout/Mul_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_3/Const�
flatten_3/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_3/Reshape�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9/Reluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_10/dropout/Const�
dropout_10/dropout/MulMuldense_9/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_10/dropout/Mul~
dropout_10/dropout/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape�
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform�
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_10/dropout/GreaterEqual/y�
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_10/dropout/GreaterEqual�
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_10/dropout/Cast�
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_10/dropout/Mul_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_10/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Const�
dropout_11/dropout/MulMuldense_10/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape�
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform�
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/y�
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_11/dropout/GreaterEqual�
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_11/dropout/Cast�
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_11/dropout/Mul_1�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentitydropout_11/dropout/Mul_1:z:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_3_input
�
h
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_706550

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
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708861

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
�
!__inference__wrapped_model_706394
input_1

cnn_706360:

cnn_706362:

cnn_706364:

cnn_706366:$

cnn_706368: 

cnn_706370: %

cnn_706372: �

cnn_706374:	�&

cnn_706376:��

cnn_706378:	�

cnn_706380:���

cnn_706382:	�

cnn_706384:
��

cnn_706386:	�

cnn_706388:	�

cnn_706390:
identity��CNN/StatefulPartitionedCall�
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1
cnn_706360
cnn_706362
cnn_706364
cnn_706366
cnn_706368
cnn_706370
cnn_706372
cnn_706374
cnn_706376
cnn_706378
cnn_706380
cnn_706382
cnn_706384
cnn_706386
cnn_706388
cnn_706390*
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
__inference_call_5898532
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
�
6__inference_batch_normalization_3_layer_call_fn_708941

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
GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7064602
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
�
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_709044

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
-__inference_sequential_3_layer_call_fn_708797
lambda_3_input
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
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2 *0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_7070722
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
_user_specified_namelambda_3_input
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_706416

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
M
1__inference_max_pooling2d_10_layer_call_fn_706544

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
GPU2 *0J 8� *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_7065382
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
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_709185

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
�
C__inference_dense_9_layer_call_and_return_conditional_losses_709100

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�
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
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
(__inference_dense_9_layer_call_fn_709109

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
GPU2 *0J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_7066922
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
�
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_709056

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
�
�
D__inference_dense_10_layer_call_and_return_conditional_losses_709159

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�s
�
__inference_call_591911

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: �E
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	�Q
5sequential_3_conv2d_11_conv2d_readvariableop_resource:��E
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	�H
3sequential_3_dense_9_matmul_readvariableop_resource:���C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�H
4sequential_3_dense_10_matmul_readvariableop_resource:
��D
5sequential_3_dense_10_biasadd_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_3/ReadVariableOp�3sequential_3/batch_normalization_3/ReadVariableOp_1�-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�,sequential_3/conv2d_10/Conv2D/ReadVariableOp�-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�,sequential_3/conv2d_11/Conv2D/ReadVariableOp�,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�+sequential_3/conv2d_9/Conv2D/ReadVariableOp�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp�
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack�
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1�
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2�
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice�
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOp�
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:�KK:::::*
epsilon%o�:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3�
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOp�
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D�
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2
sequential_3/conv2d_9/BiasAdd�
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_3/conv2d_9/Relu�
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool�
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOp�
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D�
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_3/conv2d_10/BiasAdd�
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_3/conv2d_10/Relu�
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool�
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOp�
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D�
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_3/conv2d_11/BiasAdd�
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_3/conv2d_11/Relu�
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool�
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*(
_output_shapes
:�		�2!
sequential_3/dropout_9/Identity�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*!
_output_shapes
:���2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_9/MatMul�
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_9/BiasAdd�
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_9/Relu�
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_3/dropout_10/Identity�
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_10/MatMul�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_10/BiasAdd�
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_3/dense_10/Relu�
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_3/dropout_11/Identity�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_11/BiasAddt
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_11/Softmax�
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:�KK: : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_3/batch_normalization_3/ReadVariableOp1sequential_3/batch_normalization_3/ReadVariableOp2j
3sequential_3/batch_normalization_3/ReadVariableOp_13sequential_3/batch_normalization_3/ReadVariableOp_12^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
�
-__inference_sequential_3_layer_call_fn_708731

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
GPU2 *0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_7067542
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
�
�
*__inference_conv2d_11_layer_call_fn_709039

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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_7066532
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
�
�
-__inference_sequential_3_layer_call_fn_708698
lambda_3_input
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
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2 *0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_7067542
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
_user_specified_namelambda_3_input
�
�
)__inference_conv2d_9_layer_call_fn_708999

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
GPU2 *0J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_7066172
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
�
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_706805

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_706460

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
*__inference_conv2d_10_layer_call_fn_709019

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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_7066352
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708879

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
�
M
1__inference_max_pooling2d_11_layer_call_fn_706556

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
GPU2 *0J 8� *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_7065502
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
6__inference_batch_normalization_3_layer_call_fn_708928

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
GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7064162
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
�
$__inference_CNN_layer_call_fn_708162
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
GPU2 *0J 8� *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_7073362
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
�
c
*__inference_dropout_9_layer_call_fn_709066

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
GPU2 *0J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_7068772
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
�
`
D__inference_lambda_3_layer_call_and_return_conditional_losses_706970

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
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_706673

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
�
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_709114

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
�
G
+__inference_dropout_11_layer_call_fn_709190

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
GPU2 *0J 8� *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_7067332
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
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_708478

inputs;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: C
(conv2d_10_conv2d_readvariableop_resource: �8
)conv2d_10_biasadd_readvariableop_resource:	�D
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�;
&dense_9_matmul_readvariableop_resource:���6
'dense_9_biasadd_readvariableop_resource:	�;
'dense_10_matmul_readvariableop_resource:
��7
(dense_10_biasadd_readvariableop_resource:	�
identity��$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�
lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_3/strided_slice/stack�
lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_3/strided_slice/stack_1�
lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_3/strided_slice/stack_2�
lambda_3/strided_sliceStridedSliceinputs%lambda_3/strided_slice/stack:output:0'lambda_3/strided_slice/stack_1:output:0'lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2
lambda_3/strided_slice�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3lambda_3/strided_slice:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_3/FusedBatchNormV3�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_9/Conv2D�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_9/Relu�
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2D max_pooling2d_9/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_10/Relu�
max_pooling2d_10/MaxPoolMaxPoolconv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_11/Conv2D/ReadVariableOp�
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_11/Conv2D�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_11/Relu�
max_pooling2d_11/MaxPoolMaxPoolconv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPoolw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_9/dropout/Const�
dropout_9/dropout/MulMul!max_pooling2d_11/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2
dropout_9/dropout/Mul�
dropout_9/dropout/ShapeShape!max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform�
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2"
 dropout_9/dropout/GreaterEqual/y�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2 
dropout_9/dropout/GreaterEqual�
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout_9/dropout/Cast�
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout_9/dropout/Mul_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_3/Const�
flatten_3/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_3/Reshape�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9/Reluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_10/dropout/Const�
dropout_10/dropout/MulMuldense_9/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_10/dropout/Mul~
dropout_10/dropout/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape�
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform�
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_10/dropout/GreaterEqual/y�
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_10/dropout/GreaterEqual�
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_10/dropout/Cast�
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_10/dropout/Mul_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_10/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Const�
dropout_11/dropout/MulMuldense_10/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape�
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform�
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/y�
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_11/dropout/GreaterEqual�
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_11/dropout/Cast�
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_11/dropout/Mul_1�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentitydropout_11/dropout/Mul_1:z:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
��
�
?__inference_CNN_layer_call_and_return_conditional_losses_707813

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: �E
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	�Q
5sequential_3_conv2d_11_conv2d_readvariableop_resource:��E
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	�H
3sequential_3_dense_9_matmul_readvariableop_resource:���C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�H
4sequential_3_dense_10_matmul_readvariableop_resource:
��D
5sequential_3_dense_10_biasadd_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/Square/ReadVariableOp�Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_3/ReadVariableOp�3sequential_3/batch_normalization_3/ReadVariableOp_1�-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�,sequential_3/conv2d_10/Conv2D/ReadVariableOp�-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�,sequential_3/conv2d_11/Conv2D/ReadVariableOp�,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�+sequential_3/conv2d_9/Conv2D/ReadVariableOp�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp�
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack�
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1�
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2�
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice�
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOp�
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������KK:::::*
epsilon%o�:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3�
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOp�
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D�
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp�
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/BiasAdd�
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_3/conv2d_9/Relu�
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool�
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOp�
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D�
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp�
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_3/conv2d_10/BiasAdd�
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_3/conv2d_10/Relu�
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool�
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOp�
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D�
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp�
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_3/conv2d_11/BiasAdd�
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_3/conv2d_11/Relu�
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool�
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2!
sequential_3/dropout_9/Identity�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/MatMul�
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/BiasAdd�
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_9/Relu�
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_3/dropout_10/Identity�
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/MatMul�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/BiasAdd�
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_3/dense_10/Relu�
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_3/dropout_11/Identity�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_11/Softmax�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentitydense_11/Softmax:softmax:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2�
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_3/batch_normalization_3/ReadVariableOp1sequential_3/batch_normalization_3/ReadVariableOp2j
3sequential_3/batch_normalization_3/ReadVariableOp_13sequential_3/batch_normalization_3/ReadVariableOp_12^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_706733

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
�!
"__inference__traced_restore_709579
file_prefix3
 assignvariableop_dense_11_kernel:	�.
 assignvariableop_1_dense_11_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_3_gamma:;
-assignvariableop_8_batch_normalization_3_beta:<
"assignvariableop_9_conv2d_9_kernel: /
!assignvariableop_10_conv2d_9_bias: ?
$assignvariableop_11_conv2d_10_kernel: �1
"assignvariableop_12_conv2d_10_bias:	�@
$assignvariableop_13_conv2d_11_kernel:��1
"assignvariableop_14_conv2d_11_bias:	�7
"assignvariableop_15_dense_9_kernel:���/
 assignvariableop_16_dense_9_bias:	�7
#assignvariableop_17_dense_10_kernel:
��0
!assignvariableop_18_dense_10_bias:	�C
5assignvariableop_19_batch_normalization_3_moving_mean:G
9assignvariableop_20_batch_normalization_3_moving_variance:#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_11_kernel_m:	�6
(assignvariableop_26_adam_dense_11_bias_m:D
6assignvariableop_27_adam_batch_normalization_3_gamma_m:C
5assignvariableop_28_adam_batch_normalization_3_beta_m:D
*assignvariableop_29_adam_conv2d_9_kernel_m: 6
(assignvariableop_30_adam_conv2d_9_bias_m: F
+assignvariableop_31_adam_conv2d_10_kernel_m: �8
)assignvariableop_32_adam_conv2d_10_bias_m:	�G
+assignvariableop_33_adam_conv2d_11_kernel_m:��8
)assignvariableop_34_adam_conv2d_11_bias_m:	�>
)assignvariableop_35_adam_dense_9_kernel_m:���6
'assignvariableop_36_adam_dense_9_bias_m:	�>
*assignvariableop_37_adam_dense_10_kernel_m:
��7
(assignvariableop_38_adam_dense_10_bias_m:	�=
*assignvariableop_39_adam_dense_11_kernel_v:	�6
(assignvariableop_40_adam_dense_11_bias_v:D
6assignvariableop_41_adam_batch_normalization_3_gamma_v:C
5assignvariableop_42_adam_batch_normalization_3_beta_v:D
*assignvariableop_43_adam_conv2d_9_kernel_v: 6
(assignvariableop_44_adam_conv2d_9_bias_v: F
+assignvariableop_45_adam_conv2d_10_kernel_v: �8
)assignvariableop_46_adam_conv2d_10_bias_v:	�G
+assignvariableop_47_adam_conv2d_11_kernel_v:��8
)assignvariableop_48_adam_conv2d_11_bias_v:	�>
)assignvariableop_49_adam_dense_9_kernel_v:���6
'assignvariableop_50_adam_dense_9_bias_v:	�>
*assignvariableop_51_adam_dense_10_kernel_v:
��7
(assignvariableop_52_adam_dense_10_bias_v:	�
identity_54��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*�
value�B�6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_3_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_3_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_9_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_9_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_10_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_10_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_11_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_11_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_9_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_9_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_10_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_10_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_3_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_3_moving_varianceIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_11_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_11_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_3_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_3_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_9_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_9_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_10_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_10_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_11_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_11_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_9_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_9_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_10_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_10_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_11_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_11_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_3_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_3_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_9_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_9_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_10_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_10_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_11_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_11_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_9_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_9_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_10_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_10_bias_vIdentity_52:output:0"/device:CPU:0*
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
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_706943

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
*__inference_flatten_3_layer_call_fn_709077

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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_7066732
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
E__inference_conv2d_10_layer_call_and_return_conditional_losses_709010

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
�
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_709072

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
�[
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_706754

inputs*
batch_normalization_3_706591:*
batch_normalization_3_706593:*
batch_normalization_3_706595:*
batch_normalization_3_706597:)
conv2d_9_706618: 
conv2d_9_706620: +
conv2d_10_706636: �
conv2d_10_706638:	�,
conv2d_11_706654:��
conv2d_11_706656:	�#
dense_9_706693:���
dense_9_706695:	�#
dense_10_706723:
��
dense_10_706725:	�
identity��-batch_normalization_3/StatefulPartitionedCall�!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp� dense_10/StatefulPartitionedCall�1dense_10/kernel/Regularizer/Square/ReadVariableOp�dense_9/StatefulPartitionedCall�0dense_9/kernel/Regularizer/Square/ReadVariableOp�
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8� *M
fHRF
D__inference_lambda_3_layer_call_and_return_conditional_losses_7065712
lambda_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0batch_normalization_3_706591batch_normalization_3_706593batch_normalization_3_706595batch_normalization_3_706597*
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
GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7065902/
-batch_normalization_3/StatefulPartitionedCall�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_9_706618conv2d_9_706620*
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
GPU2 *0J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_7066172"
 conv2d_9/StatefulPartitionedCall�
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_7065262!
max_pooling2d_9/PartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0conv2d_10_706636conv2d_10_706638*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_7066352#
!conv2d_10/StatefulPartitionedCall�
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_7065382"
 max_pooling2d_10/PartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_706654conv2d_11_706656*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_7066532#
!conv2d_11/StatefulPartitionedCall�
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_7065502"
 max_pooling2d_11/PartitionedCall�
dropout_9/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
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
GPU2 *0J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_7066652
dropout_9/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_7066732
flatten_3/PartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_706693dense_9_706695*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_7066922!
dense_9/StatefulPartitionedCall�
dropout_10/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_7067032
dropout_10/PartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_10_706723dense_10_706725*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_7067222"
 dense_10/StatefulPartitionedCall�
dropout_11/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_7067332
dropout_11/PartitionedCall�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_706618*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_706693*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_706723* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentity#dropout_11/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_9/StatefulPartitionedCall1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_709228N
:dense_10_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_10/kernel/Regularizer/Square/ReadVariableOp�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_10_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentity#dense_10/kernel/Regularizer/mul:z:02^dense_10/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp
�1
�
?__inference_CNN_layer_call_and_return_conditional_losses_707336

inputs!
sequential_3_707271:!
sequential_3_707273:!
sequential_3_707275:!
sequential_3_707277:-
sequential_3_707279: !
sequential_3_707281: .
sequential_3_707283: �"
sequential_3_707285:	�/
sequential_3_707287:��"
sequential_3_707289:	�(
sequential_3_707291:���"
sequential_3_707293:	�'
sequential_3_707295:
��"
sequential_3_707297:	�"
dense_11_707312:	�
dense_11_707314:
identity��1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp� dense_11/StatefulPartitionedCall�0dense_9/kernel/Regularizer/Square/ReadVariableOp�$sequential_3/StatefulPartitionedCall�
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_707271sequential_3_707273sequential_3_707275sequential_3_707277sequential_3_707279sequential_3_707281sequential_3_707283sequential_3_707285sequential_3_707287sequential_3_707289sequential_3_707291sequential_3_707293sequential_3_707295sequential_3_707297*
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
GPU2 *0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_7067542&
$sequential_3/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0dense_11_707312dense_11_707314*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_7073112"
 dense_11/StatefulPartitionedCall�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_707279*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_707291*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_707295* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp!^dense_11/StatefulPartitionedCall1^dense_9/kernel/Regularizer/Square/ReadVariableOp%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_706665

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
�1
�
?__inference_CNN_layer_call_and_return_conditional_losses_707476

inputs!
sequential_3_707423:!
sequential_3_707425:!
sequential_3_707427:!
sequential_3_707429:-
sequential_3_707431: !
sequential_3_707433: .
sequential_3_707435: �"
sequential_3_707437:	�/
sequential_3_707439:��"
sequential_3_707441:	�(
sequential_3_707443:���"
sequential_3_707445:	�'
sequential_3_707447:
��"
sequential_3_707449:	�"
dense_11_707452:	�
dense_11_707454:
identity��1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�1dense_10/kernel/Regularizer/Square/ReadVariableOp� dense_11/StatefulPartitionedCall�0dense_9/kernel/Regularizer/Square/ReadVariableOp�$sequential_3/StatefulPartitionedCall�
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_707423sequential_3_707425sequential_3_707427sequential_3_707429sequential_3_707431sequential_3_707433sequential_3_707435sequential_3_707437sequential_3_707439sequential_3_707441sequential_3_707443sequential_3_707445sequential_3_707447sequential_3_707449*
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
GPU2 *0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_7070722&
$sequential_3/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0dense_11_707452dense_11_707454*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_7073112"
 dense_11/StatefulPartitionedCall�
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_707431*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/Square�
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const�
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum�
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!conv2d_9/kernel/Regularizer/mul/x�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul�
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_707443*!
_output_shapes
:���*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp�
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2#
!dense_9/kernel/Regularizer/Square�
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const�
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/Sum�
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_9/kernel/Regularizer/mul/x�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul�
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_707447* 
_output_shapes
:
��*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp�
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_10/kernel/Regularizer/Square�
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const�
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/Sum�
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_10/kernel/Regularizer/mul/x�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp!^dense_11/StatefulPartitionedCall1^dense_9/kernel/Regularizer/Square/ReadVariableOp%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_3_layer_call_fn_708967

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
GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7069432
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
�
`
D__inference_lambda_3_layer_call_and_return_conditional_losses_708833

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
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_706590

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
6__inference_batch_normalization_3_layer_call_fn_708954

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
GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7065902
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:ă
�


h2ptjl
_output
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature
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
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�d
_tf_keras_sequential�d{"name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_3_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_3_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_3_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem�m�&m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�v�v�&v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�"
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
12
13"
trackable_list_wrapper
�
&0
'1
22
33
(4
)5
*6
+7
,8
-9
.10
/11
012
113
14
15"
trackable_list_wrapper
�
regularization_losses
4metrics
trainable_variables
5layer_regularization_losses

6layers
7layer_metrics
8non_trainable_variables
	variables
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
:trainable_variables
;	variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "lambda_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
�

=axis
	&gamma
'beta
2moving_mean
3moving_variance
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

(kernel
)bias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
�


*kernel
+bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
�
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
�


,kernel
-bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
�
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
�
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
�
^regularization_losses
_trainable_variables
`	variables
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
�	

.kernel
/bias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
�
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
�	

0kernel
1bias
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
8
�0
�1
�2"
trackable_list_wrapper
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
111"
trackable_list_wrapper
�
&0
'1
22
33
(4
)5
*6
+7
,8
-9
.10
/11
012
113"
trackable_list_wrapper
�
regularization_losses
rmetrics
trainable_variables
slayer_regularization_losses

tlayers
ulayer_metrics
vnon_trainable_variables
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_11/kernel
:2dense_11/bias
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
regularization_losses
wmetrics
trainable_variables
xlayer_regularization_losses

ylayers
zlayer_metrics
{non_trainable_variables
	variables
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
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
):' 2conv2d_9/kernel
: 2conv2d_9/bias
+:) �2conv2d_10/kernel
:�2conv2d_10/bias
,:*��2conv2d_11/kernel
:�2conv2d_11/bias
#:!���2dense_9/kernel
:�2dense_9/bias
#:!
��2dense_10/kernel
:�2dense_10/bias
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
9regularization_losses
~metrics
:trainable_variables
layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
;	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
<
&0
'1
22
33"
trackable_list_wrapper
�
>regularization_losses
�metrics
?trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
@	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
Bregularization_losses
�metrics
Ctrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
D	variables
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
Fregularization_losses
�metrics
Gtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
H	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
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
Jregularization_losses
�metrics
Ktrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
L	variables
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
Nregularization_losses
�metrics
Otrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
P	variables
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
Rregularization_losses
�metrics
Strainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
T	variables
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
Vregularization_losses
�metrics
Wtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
X	variables
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
Zregularization_losses
�metrics
[trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
\	variables
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
^regularization_losses
�metrics
_trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
`	variables
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
bregularization_losses
�metrics
ctrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
d	variables
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
fregularization_losses
�metrics
gtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
h	variables
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
jregularization_losses
�metrics
ktrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
l	variables
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
nregularization_losses
�metrics
otrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
p	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
.
20
31"
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
.
20
31"
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
':%	�2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
.:,2"Adam/batch_normalization_3/gamma/m
-:+2!Adam/batch_normalization_3/beta/m
.:, 2Adam/conv2d_9/kernel/m
 : 2Adam/conv2d_9/bias/m
0:. �2Adam/conv2d_10/kernel/m
": �2Adam/conv2d_10/bias/m
1:/��2Adam/conv2d_11/kernel/m
": �2Adam/conv2d_11/bias/m
(:&���2Adam/dense_9/kernel/m
 :�2Adam/dense_9/bias/m
(:&
��2Adam/dense_10/kernel/m
!:�2Adam/dense_10/bias/m
':%	�2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
.:,2"Adam/batch_normalization_3/gamma/v
-:+2!Adam/batch_normalization_3/beta/v
.:, 2Adam/conv2d_9/kernel/v
 : 2Adam/conv2d_9/bias/v
0:. �2Adam/conv2d_10/kernel/v
": �2Adam/conv2d_10/bias/v
1:/��2Adam/conv2d_11/kernel/v
": �2Adam/conv2d_11/bias/v
(:&���2Adam/dense_9/kernel/v
 :�2Adam/dense_9/bias/v
(:&
��2Adam/dense_10/kernel/v
!:�2Adam/dense_10/bias/v
�2�
?__inference_CNN_layer_call_and_return_conditional_losses_707813
?__inference_CNN_layer_call_and_return_conditional_losses_707924
?__inference_CNN_layer_call_and_return_conditional_losses_708014
?__inference_CNN_layer_call_and_return_conditional_losses_708125�
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
$__inference_CNN_layer_call_fn_708162
$__inference_CNN_layer_call_fn_708199
$__inference_CNN_layer_call_fn_708236
$__inference_CNN_layer_call_fn_708273�
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
!__inference__wrapped_model_706394�
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
__inference_call_591839
__inference_call_591911
__inference_call_591983�
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
H__inference_sequential_3_layer_call_and_return_conditional_losses_708374
H__inference_sequential_3_layer_call_and_return_conditional_losses_708478
H__inference_sequential_3_layer_call_and_return_conditional_losses_708561
H__inference_sequential_3_layer_call_and_return_conditional_losses_708665�
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
�2�
-__inference_sequential_3_layer_call_fn_708698
-__inference_sequential_3_layer_call_fn_708731
-__inference_sequential_3_layer_call_fn_708764
-__inference_sequential_3_layer_call_fn_708797�
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
D__inference_dense_11_layer_call_and_return_conditional_losses_708808�
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
)__inference_dense_11_layer_call_fn_708817�
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
$__inference_signature_wrapper_707723input_1"�
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
D__inference_lambda_3_layer_call_and_return_conditional_losses_708825
D__inference_lambda_3_layer_call_and_return_conditional_losses_708833�
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
)__inference_lambda_3_layer_call_fn_708838
)__inference_lambda_3_layer_call_fn_708843�
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708861
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708879
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708897
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708915�
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
6__inference_batch_normalization_3_layer_call_fn_708928
6__inference_batch_normalization_3_layer_call_fn_708941
6__inference_batch_normalization_3_layer_call_fn_708954
6__inference_batch_normalization_3_layer_call_fn_708967�
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
D__inference_conv2d_9_layer_call_and_return_conditional_losses_708990�
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
)__inference_conv2d_9_layer_call_fn_708999�
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
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_706526�
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
0__inference_max_pooling2d_9_layer_call_fn_706532�
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
E__inference_conv2d_10_layer_call_and_return_conditional_losses_709010�
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
*__inference_conv2d_10_layer_call_fn_709019�
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
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_706538�
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
1__inference_max_pooling2d_10_layer_call_fn_706544�
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
E__inference_conv2d_11_layer_call_and_return_conditional_losses_709030�
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
*__inference_conv2d_11_layer_call_fn_709039�
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
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_706550�
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
1__inference_max_pooling2d_11_layer_call_fn_706556�
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
E__inference_dropout_9_layer_call_and_return_conditional_losses_709044
E__inference_dropout_9_layer_call_and_return_conditional_losses_709056�
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
*__inference_dropout_9_layer_call_fn_709061
*__inference_dropout_9_layer_call_fn_709066�
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_709072�
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
*__inference_flatten_3_layer_call_fn_709077�
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
C__inference_dense_9_layer_call_and_return_conditional_losses_709100�
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
(__inference_dense_9_layer_call_fn_709109�
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
F__inference_dropout_10_layer_call_and_return_conditional_losses_709114
F__inference_dropout_10_layer_call_and_return_conditional_losses_709126�
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
+__inference_dropout_10_layer_call_fn_709131
+__inference_dropout_10_layer_call_fn_709136�
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
D__inference_dense_10_layer_call_and_return_conditional_losses_709159�
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
)__inference_dense_10_layer_call_fn_709168�
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
F__inference_dropout_11_layer_call_and_return_conditional_losses_709173
F__inference_dropout_11_layer_call_and_return_conditional_losses_709185�
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
+__inference_dropout_11_layer_call_fn_709190
+__inference_dropout_11_layer_call_fn_709195�
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
__inference_loss_fn_0_709206�
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
__inference_loss_fn_1_709217�
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
__inference_loss_fn_2_709228�
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
?__inference_CNN_layer_call_and_return_conditional_losses_707813v&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p 
� "%�"
�
0���������
� �
?__inference_CNN_layer_call_and_return_conditional_losses_707924v&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p
� "%�"
�
0���������
� �
?__inference_CNN_layer_call_and_return_conditional_losses_708014w&'23()*+,-./01<�9
2�/
)�&
input_1���������KK
p 
� "%�"
�
0���������
� �
?__inference_CNN_layer_call_and_return_conditional_losses_708125w&'23()*+,-./01<�9
2�/
)�&
input_1���������KK
p
� "%�"
�
0���������
� �
$__inference_CNN_layer_call_fn_708162j&'23()*+,-./01<�9
2�/
)�&
input_1���������KK
p 
� "�����������
$__inference_CNN_layer_call_fn_708199i&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p 
� "�����������
$__inference_CNN_layer_call_fn_708236i&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p
� "�����������
$__inference_CNN_layer_call_fn_708273j&'23()*+,-./01<�9
2�/
)�&
input_1���������KK
p
� "�����������
!__inference__wrapped_model_706394�&'23()*+,-./018�5
.�+
)�&
input_1���������KK
� "3�0
.
output_1"�
output_1����������
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708861�&'23M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708879�&'23M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708897r&'23;�8
1�.
(�%
inputs���������KK
p 
� "-�*
#� 
0���������KK
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_708915r&'23;�8
1�.
(�%
inputs���������KK
p
� "-�*
#� 
0���������KK
� �
6__inference_batch_normalization_3_layer_call_fn_708928�&'23M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
6__inference_batch_normalization_3_layer_call_fn_708941�&'23M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
6__inference_batch_normalization_3_layer_call_fn_708954e&'23;�8
1�.
(�%
inputs���������KK
p 
� " ����������KK�
6__inference_batch_normalization_3_layer_call_fn_708967e&'23;�8
1�.
(�%
inputs���������KK
p
� " ����������KKt
__inference_call_591839Y&'23()*+,-./013�0
)�&
 �
inputs�KK
p
� "�	�t
__inference_call_591911Y&'23()*+,-./013�0
)�&
 �
inputs�KK
p 
� "�	��
__inference_call_591983i&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p 
� "�����������
E__inference_conv2d_10_layer_call_and_return_conditional_losses_709010m*+7�4
-�*
(�%
inputs���������%% 
� ".�+
$�!
0���������%%�
� �
*__inference_conv2d_10_layer_call_fn_709019`*+7�4
-�*
(�%
inputs���������%% 
� "!����������%%��
E__inference_conv2d_11_layer_call_and_return_conditional_losses_709030n,-8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
*__inference_conv2d_11_layer_call_fn_709039a,-8�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv2d_9_layer_call_and_return_conditional_losses_708990l()7�4
-�*
(�%
inputs���������KK
� "-�*
#� 
0���������KK 
� �
)__inference_conv2d_9_layer_call_fn_708999_()7�4
-�*
(�%
inputs���������KK
� " ����������KK �
D__inference_dense_10_layer_call_and_return_conditional_losses_709159^010�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_10_layer_call_fn_709168Q010�-
&�#
!�
inputs����������
� "������������
D__inference_dense_11_layer_call_and_return_conditional_losses_708808]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
)__inference_dense_11_layer_call_fn_708817P0�-
&�#
!�
inputs����������
� "�����������
C__inference_dense_9_layer_call_and_return_conditional_losses_709100_./1�.
'�$
"�
inputs�����������
� "&�#
�
0����������
� ~
(__inference_dense_9_layer_call_fn_709109R./1�.
'�$
"�
inputs�����������
� "������������
F__inference_dropout_10_layer_call_and_return_conditional_losses_709114^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_dropout_10_layer_call_and_return_conditional_losses_709126^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
+__inference_dropout_10_layer_call_fn_709131Q4�1
*�'
!�
inputs����������
p 
� "������������
+__inference_dropout_10_layer_call_fn_709136Q4�1
*�'
!�
inputs����������
p
� "������������
F__inference_dropout_11_layer_call_and_return_conditional_losses_709173^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_dropout_11_layer_call_and_return_conditional_losses_709185^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
+__inference_dropout_11_layer_call_fn_709190Q4�1
*�'
!�
inputs����������
p 
� "������������
+__inference_dropout_11_layer_call_fn_709195Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_dropout_9_layer_call_and_return_conditional_losses_709044n<�9
2�/
)�&
inputs���������		�
p 
� ".�+
$�!
0���������		�
� �
E__inference_dropout_9_layer_call_and_return_conditional_losses_709056n<�9
2�/
)�&
inputs���������		�
p
� ".�+
$�!
0���������		�
� �
*__inference_dropout_9_layer_call_fn_709061a<�9
2�/
)�&
inputs���������		�
p 
� "!����������		��
*__inference_dropout_9_layer_call_fn_709066a<�9
2�/
)�&
inputs���������		�
p
� "!����������		��
E__inference_flatten_3_layer_call_and_return_conditional_losses_709072c8�5
.�+
)�&
inputs���������		�
� "'�$
�
0�����������
� �
*__inference_flatten_3_layer_call_fn_709077V8�5
.�+
)�&
inputs���������		�
� "�������������
D__inference_lambda_3_layer_call_and_return_conditional_losses_708825p?�<
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
D__inference_lambda_3_layer_call_and_return_conditional_losses_708833p?�<
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
)__inference_lambda_3_layer_call_fn_708838c?�<
5�2
(�%
inputs���������KK

 
p 
� " ����������KK�
)__inference_lambda_3_layer_call_fn_708843c?�<
5�2
(�%
inputs���������KK

 
p
� " ����������KK;
__inference_loss_fn_0_709206(�

� 
� "� ;
__inference_loss_fn_1_709217.�

� 
� "� ;
__inference_loss_fn_2_7092280�

� 
� "� �
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_706538�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_10_layer_call_fn_706544�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_706550�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_11_layer_call_fn_706556�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_706526�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_max_pooling2d_9_layer_call_fn_706532�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
H__inference_sequential_3_layer_call_and_return_conditional_losses_708374y&'23()*+,-./01?�<
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
H__inference_sequential_3_layer_call_and_return_conditional_losses_708478y&'23()*+,-./01?�<
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
H__inference_sequential_3_layer_call_and_return_conditional_losses_708561�&'23()*+,-./01G�D
=�:
0�-
lambda_3_input���������KK
p 

 
� "&�#
�
0����������
� �
H__inference_sequential_3_layer_call_and_return_conditional_losses_708665�&'23()*+,-./01G�D
=�:
0�-
lambda_3_input���������KK
p

 
� "&�#
�
0����������
� �
-__inference_sequential_3_layer_call_fn_708698t&'23()*+,-./01G�D
=�:
0�-
lambda_3_input���������KK
p 

 
� "������������
-__inference_sequential_3_layer_call_fn_708731l&'23()*+,-./01?�<
5�2
(�%
inputs���������KK
p 

 
� "������������
-__inference_sequential_3_layer_call_fn_708764l&'23()*+,-./01?�<
5�2
(�%
inputs���������KK
p

 
� "������������
-__inference_sequential_3_layer_call_fn_708797t&'23()*+,-./01G�D
=�:
0�-
lambda_3_input���������KK
p

 
� "������������
$__inference_signature_wrapper_707723�&'23()*+,-./01C�@
� 
9�6
4
input_1)�&
input_1���������KK"3�0
.
output_1"�
output_1���������