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
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ڧ
{
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_29/kernel
t
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes
:	�*
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
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
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
: *
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
: *
dtype0
�
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*!
shared_nameconv2d_28/kernel
~
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*'
_output_shapes
: �*
dtype0
u
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_28/bias
n
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes	
:�*
dtype0
�
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_29/kernel

$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_29/bias
n
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes	
:�*
dtype0
}
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���* 
shared_namedense_27/kernel
v
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*!
_output_shapes
:���*
dtype0
s
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_27/bias
l
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes	
:�*
dtype0
|
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_28/kernel
u
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel* 
_output_shapes
:
��*
dtype0
s
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_28/bias
l
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes	
:�*
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
Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_29/kernel/m
�
*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/m
y
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
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
Adam/conv2d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_27/kernel/m
�
+Adam/conv2d_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2d_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_27/bias/m
{
)Adam/conv2d_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_28/kernel/m
�
+Adam/conv2d_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/m*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_28/bias/m
|
)Adam/conv2d_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_29/kernel/m
�
+Adam/conv2d_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_29/bias/m
|
)Adam/conv2d_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*'
shared_nameAdam/dense_27/kernel/m
�
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*!
_output_shapes
:���*
dtype0
�
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_27/bias/m
z
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_28/kernel/m
�
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_28/bias/m
z
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_29/kernel/v
�
*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/v
y
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
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
Adam/conv2d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_27/kernel/v
�
+Adam/conv2d_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2d_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_27/bias/v
{
)Adam/conv2d_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*(
shared_nameAdam/conv2d_28/kernel/v
�
+Adam/conv2d_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/v*'
_output_shapes
: �*
dtype0
�
Adam/conv2d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_28/bias/v
|
)Adam/conv2d_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_29/kernel/v
�
+Adam/conv2d_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_29/bias/v
|
)Adam/conv2d_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*'
shared_nameAdam/dense_27/kernel/v
�
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*!
_output_shapes
:���*
dtype0
�
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_27/bias/v
z
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_28/kernel/v
�
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_28/bias/v
z
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�`
value�`B�` B�_
�

h2ptjl
_output
	optimizer
trainable_variables
	variables
regularization_losses
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
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem�m�&m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�v�v�&v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�
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
 
�
4non_trainable_variables
trainable_variables
5layer_metrics
	variables
6layer_regularization_losses

7layers
8metrics
regularization_losses
 
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
�
=axis
	&gamma
'beta
2moving_mean
3moving_variance
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

(kernel
)bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
R
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
h

*kernel
+bias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
R
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
h

,kernel
-bias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
R
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
R
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
R
^trainable_variables
_	variables
`regularization_losses
a	keras_api
h

.kernel
/bias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
R
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
h

0kernel
1bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
R
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
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
 
�
rnon_trainable_variables
trainable_variables
slayer_metrics
	variables
tlayer_regularization_losses

ulayers
vmetrics
regularization_losses
NL
VARIABLE_VALUEdense_29/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_29/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
wnon_trainable_variables
trainable_variables
xlayer_metrics
	variables
ylayer_regularization_losses

zlayers
{metrics
regularization_losses
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
VARIABLE_VALUEbatch_normalization_9/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_9/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_27/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_27/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_28/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_28/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_29/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_29/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_27/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_27/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_28/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_28/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_9/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_9/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE

20
31
 
 

0
1

|0
}1
 
 
 
�
~non_trainable_variables
9trainable_variables
layer_metrics
:	variables
 �layer_regularization_losses
�layers
�metrics
;regularization_losses
 

&0
'1

&0
'1
22
33
 
�
�non_trainable_variables
>trainable_variables
�layer_metrics
?	variables
 �layer_regularization_losses
�layers
�metrics
@regularization_losses

(0
)1

(0
)1
 
�
�non_trainable_variables
Btrainable_variables
�layer_metrics
C	variables
 �layer_regularization_losses
�layers
�metrics
Dregularization_losses
 
 
 
�
�non_trainable_variables
Ftrainable_variables
�layer_metrics
G	variables
 �layer_regularization_losses
�layers
�metrics
Hregularization_losses

*0
+1

*0
+1
 
�
�non_trainable_variables
Jtrainable_variables
�layer_metrics
K	variables
 �layer_regularization_losses
�layers
�metrics
Lregularization_losses
 
 
 
�
�non_trainable_variables
Ntrainable_variables
�layer_metrics
O	variables
 �layer_regularization_losses
�layers
�metrics
Pregularization_losses

,0
-1

,0
-1
 
�
�non_trainable_variables
Rtrainable_variables
�layer_metrics
S	variables
 �layer_regularization_losses
�layers
�metrics
Tregularization_losses
 
 
 
�
�non_trainable_variables
Vtrainable_variables
�layer_metrics
W	variables
 �layer_regularization_losses
�layers
�metrics
Xregularization_losses
 
 
 
�
�non_trainable_variables
Ztrainable_variables
�layer_metrics
[	variables
 �layer_regularization_losses
�layers
�metrics
\regularization_losses
 
 
 
�
�non_trainable_variables
^trainable_variables
�layer_metrics
_	variables
 �layer_regularization_losses
�layers
�metrics
`regularization_losses

.0
/1

.0
/1
 
�
�non_trainable_variables
btrainable_variables
�layer_metrics
c	variables
 �layer_regularization_losses
�layers
�metrics
dregularization_losses
 
 
 
�
�non_trainable_variables
ftrainable_variables
�layer_metrics
g	variables
 �layer_regularization_losses
�layers
�metrics
hregularization_losses

00
11

00
11
 
�
�non_trainable_variables
jtrainable_variables
�layer_metrics
k	variables
 �layer_regularization_losses
�layers
�metrics
lregularization_losses
 
 
 
�
�non_trainable_variables
ntrainable_variables
�layer_metrics
o	variables
 �layer_regularization_losses
�layers
�metrics
pregularization_losses

20
31
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
VARIABLE_VALUEAdam/dense_29/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_29/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_9/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_27/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_27/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_28/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_28/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_29/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_29/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_27/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_27/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_28/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_28/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_29/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_29/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_9/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_27/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_27/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_28/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_28/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_29/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_29/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_27/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_27/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_28/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_28/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������KK*
dtype0*$
shape:���������KK
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias*
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
%__inference_signature_wrapper_1909877
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp+Adam/conv2d_27/kernel/m/Read/ReadVariableOp)Adam/conv2d_27/bias/m/Read/ReadVariableOp+Adam/conv2d_28/kernel/m/Read/ReadVariableOp)Adam/conv2d_28/bias/m/Read/ReadVariableOp+Adam/conv2d_29/kernel/m/Read/ReadVariableOp)Adam/conv2d_29/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp+Adam/conv2d_27/kernel/v/Read/ReadVariableOp)Adam/conv2d_27/bias/v/Read/ReadVariableOp+Adam/conv2d_28/kernel/v/Read/ReadVariableOp)Adam/conv2d_28/bias/v/Read/ReadVariableOp+Adam/conv2d_29/kernel/v/Read/ReadVariableOp)Adam/conv2d_29/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOpConst*B
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
 __inference__traced_save_1911564
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_29/kerneldense_29/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_9/gammabatch_normalization_9/betaconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/bias!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancetotalcounttotal_1count_1Adam/dense_29/kernel/mAdam/dense_29/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/conv2d_27/kernel/mAdam/conv2d_27/bias/mAdam/conv2d_28/kernel/mAdam/conv2d_28/bias/mAdam/conv2d_29/kernel/mAdam/conv2d_29/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/vAdam/dense_29/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/conv2d_27/kernel/vAdam/conv2d_27/bias/vAdam/conv2d_28/kernel/vAdam/conv2d_28/bias/vAdam/conv2d_29/kernel/vAdam/conv2d_29/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/v*A
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
#__inference__traced_restore_1911733�
�a
�	
I__inference_sequential_9_layer_call_and_return_conditional_losses_1909226

inputs+
batch_normalization_9_1909166:+
batch_normalization_9_1909168:+
batch_normalization_9_1909170:+
batch_normalization_9_1909172:+
conv2d_27_1909175: 
conv2d_27_1909177: ,
conv2d_28_1909181: � 
conv2d_28_1909183:	�-
conv2d_29_1909187:�� 
conv2d_29_1909189:	�%
dense_27_1909195:���
dense_27_1909197:	�$
dense_28_1909201:
��
dense_28_1909203:	�
identity��-batch_normalization_9/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�1dense_27/kernel/Regularizer/Square/ReadVariableOp� dense_28/StatefulPartitionedCall�1dense_28/kernel/Regularizer/Square/ReadVariableOp�"dropout_27/StatefulPartitionedCall�"dropout_28/StatefulPartitionedCall�"dropout_29/StatefulPartitionedCall�
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
E__inference_lambda_9_layer_call_and_return_conditional_losses_19091242
lambda_9/PartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall!lambda_9/PartitionedCall:output:0batch_normalization_9_1909166batch_normalization_9_1909168batch_normalization_9_1909170batch_normalization_9_1909172*
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_19090972/
-batch_normalization_9/StatefulPartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_27_1909175conv2d_27_1909177*
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_19087712#
!conv2d_27/StatefulPartitionedCall�
 max_pooling2d_27/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_19086802"
 max_pooling2d_27/PartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_27/PartitionedCall:output:0conv2d_28_1909181conv2d_28_1909183*
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
F__inference_conv2d_28_layer_call_and_return_conditional_losses_19087892#
!conv2d_28/StatefulPartitionedCall�
 max_pooling2d_28/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_19086922"
 max_pooling2d_28/PartitionedCall�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_28/PartitionedCall:output:0conv2d_29_1909187conv2d_29_1909189*
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
F__inference_conv2d_29_layer_call_and_return_conditional_losses_19088072#
!conv2d_29/StatefulPartitionedCall�
 max_pooling2d_29/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_19087042"
 max_pooling2d_29/PartitionedCall�
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_29/PartitionedCall:output:0*
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
G__inference_dropout_27_layer_call_and_return_conditional_losses_19090312$
"dropout_27/StatefulPartitionedCall�
flatten_9/PartitionedCallPartitionedCall+dropout_27/StatefulPartitionedCall:output:0*
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
F__inference_flatten_9_layer_call_and_return_conditional_losses_19088272
flatten_9/PartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_27_1909195dense_27_1909197*
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
E__inference_dense_27_layer_call_and_return_conditional_losses_19088462"
 dense_27/StatefulPartitionedCall�
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0#^dropout_27/StatefulPartitionedCall*
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
G__inference_dropout_28_layer_call_and_return_conditional_losses_19089922$
"dropout_28/StatefulPartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0dense_28_1909201dense_28_1909203*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_19088762"
 dense_28/StatefulPartitionedCall�
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
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
G__inference_dropout_29_layer_call_and_return_conditional_losses_19089592$
"dropout_29/StatefulPartitionedCall�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_1909175*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_27_1909195*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_28_1909201* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentity+dropout_29/StatefulPartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall2^dense_27/kernel/Regularizer/Square/ReadVariableOp!^dense_28/StatefulPartitionedCall2^dense_28/kernel/Regularizer/Square/ReadVariableOp#^dropout_27/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
E__inference_dense_27_layer_call_and_return_conditional_losses_1911254

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
f
G__inference_dropout_29_layer_call_and_return_conditional_losses_1911339

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
%__inference_signature_wrapper_1909877
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
"__inference__wrapped_model_19085482
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
*__inference_dense_29_layer_call_fn_1910971

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
E__inference_dense_29_layer_call_and_return_conditional_losses_19094652
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
�
�
"__inference__wrapped_model_1908548
input_1
cnn_1908514:
cnn_1908516:
cnn_1908518:
cnn_1908520:%
cnn_1908522: 
cnn_1908524: &
cnn_1908526: �
cnn_1908528:	�'
cnn_1908530:��
cnn_1908532:	� 
cnn_1908534:���
cnn_1908536:	�
cnn_1908538:
��
cnn_1908540:	�
cnn_1908542:	�
cnn_1908544:
identity��CNN/StatefulPartitionedCall�
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn_1908514cnn_1908516cnn_1908518cnn_1908520cnn_1908522cnn_1908524cnn_1908526cnn_1908528cnn_1908530cnn_1908532cnn_1908534cnn_1908536cnn_1908538cnn_1908540cnn_1908542cnn_1908544*
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
GPU2 *0J 8� *!
fR
__inference_call_17887812
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
��
�
@__inference_CNN_layer_call_and_return_conditional_losses_1910168
input_1H
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_27_conv2d_readvariableop_resource: D
6sequential_9_conv2d_27_biasadd_readvariableop_resource: P
5sequential_9_conv2d_28_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_28_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_29_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_29_biasadd_readvariableop_resource:	�I
4sequential_9_dense_27_matmul_readvariableop_resource:���D
5sequential_9_dense_27_biasadd_readvariableop_resource:	�H
4sequential_9_dense_28_matmul_readvariableop_resource:
��D
5sequential_9_dense_28_biasadd_readvariableop_resource:	�:
'dense_29_matmul_readvariableop_resource:	�6
(dense_29_biasadd_readvariableop_resource:
identity��2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�,sequential_9/conv2d_27/Conv2D/ReadVariableOp�-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�,sequential_9/conv2d_28/Conv2D/ReadVariableOp�-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�,sequential_9/conv2d_29/Conv2D/ReadVariableOp�,sequential_9/dense_27/BiasAdd/ReadVariableOp�+sequential_9/dense_27/MatMul/ReadVariableOp�,sequential_9/dense_28/BiasAdd/ReadVariableOp�+sequential_9/dense_28/MatMul/ReadVariableOp�
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
,sequential_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_27/Conv2D/ReadVariableOp�
sequential_9/conv2d_27/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_27/Conv2D�
-sequential_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�
sequential_9/conv2d_27/BiasAddBiasAdd&sequential_9/conv2d_27/Conv2D:output:05sequential_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_27/BiasAdd�
sequential_9/conv2d_27/ReluRelu'sequential_9/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_27/Relu�
%sequential_9/max_pooling2d_27/MaxPoolMaxPool)sequential_9/conv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_27/MaxPool�
,sequential_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_28/Conv2D/ReadVariableOp�
sequential_9/conv2d_28/Conv2DConv2D.sequential_9/max_pooling2d_27/MaxPool:output:04sequential_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_28/Conv2D�
-sequential_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�
sequential_9/conv2d_28/BiasAddBiasAdd&sequential_9/conv2d_28/Conv2D:output:05sequential_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_28/BiasAdd�
sequential_9/conv2d_28/ReluRelu'sequential_9/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_28/Relu�
%sequential_9/max_pooling2d_28/MaxPoolMaxPool)sequential_9/conv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_28/MaxPool�
,sequential_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_29/Conv2D/ReadVariableOp�
sequential_9/conv2d_29/Conv2DConv2D.sequential_9/max_pooling2d_28/MaxPool:output:04sequential_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_29/Conv2D�
-sequential_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�
sequential_9/conv2d_29/BiasAddBiasAdd&sequential_9/conv2d_29/Conv2D:output:05sequential_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_29/BiasAdd�
sequential_9/conv2d_29/ReluRelu'sequential_9/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_29/Relu�
%sequential_9/max_pooling2d_29/MaxPoolMaxPool)sequential_9/conv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_29/MaxPool�
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_29/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_9/dropout_27/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp�
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/MatMul�
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp�
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/BiasAdd�
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/Relu�
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_28/Identity�
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp�
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/MatMul�
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp�
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/BiasAdd�
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/Relu�
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_29/Identity�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_29/Softmax�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentitydense_29/Softmax:softmax:03^conv2d_27/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_27/BiasAdd/ReadVariableOp-^sequential_9/conv2d_27/Conv2D/ReadVariableOp.^sequential_9/conv2d_28/BiasAdd/ReadVariableOp-^sequential_9/conv2d_28/Conv2D/ReadVariableOp.^sequential_9/conv2d_29/BiasAdd/ReadVariableOp-^sequential_9/conv2d_29/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp-sequential_9/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_27/Conv2D/ReadVariableOp,sequential_9/conv2d_27/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp-sequential_9/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_28/Conv2D/ReadVariableOp,sequential_9/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp-sequential_9/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_29/Conv2D/ReadVariableOp,sequential_9/conv2d_29/Conv2D/ReadVariableOp2\
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
F__inference_conv2d_27_layer_call_and_return_conditional_losses_1908771

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
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
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
f
G__inference_dropout_28_layer_call_and_return_conditional_losses_1908992

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
+__inference_conv2d_29_layer_call_fn_1911193

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
F__inference_conv2d_29_layer_call_and_return_conditional_losses_19088072
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
�
�
7__inference_batch_normalization_9_layer_call_fn_1911121

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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_19090972
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
�s
�
__inference_call_1790839

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_27_conv2d_readvariableop_resource: D
6sequential_9_conv2d_27_biasadd_readvariableop_resource: P
5sequential_9_conv2d_28_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_28_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_29_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_29_biasadd_readvariableop_resource:	�I
4sequential_9_dense_27_matmul_readvariableop_resource:���D
5sequential_9_dense_27_biasadd_readvariableop_resource:	�H
4sequential_9_dense_28_matmul_readvariableop_resource:
��D
5sequential_9_dense_28_biasadd_readvariableop_resource:	�:
'dense_29_matmul_readvariableop_resource:	�6
(dense_29_biasadd_readvariableop_resource:
identity��dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�,sequential_9/conv2d_27/Conv2D/ReadVariableOp�-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�,sequential_9/conv2d_28/Conv2D/ReadVariableOp�-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�,sequential_9/conv2d_29/Conv2D/ReadVariableOp�,sequential_9/dense_27/BiasAdd/ReadVariableOp�+sequential_9/dense_27/MatMul/ReadVariableOp�,sequential_9/dense_28/BiasAdd/ReadVariableOp�+sequential_9/dense_28/MatMul/ReadVariableOp�
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
,sequential_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_27/Conv2D/ReadVariableOp�
sequential_9/conv2d_27/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_9/conv2d_27/Conv2D�
-sequential_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�
sequential_9/conv2d_27/BiasAddBiasAdd&sequential_9/conv2d_27/Conv2D:output:05sequential_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_9/conv2d_27/BiasAdd�
sequential_9/conv2d_27/ReluRelu'sequential_9/conv2d_27/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_9/conv2d_27/Relu�
%sequential_9/max_pooling2d_27/MaxPoolMaxPool)sequential_9/conv2d_27/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_27/MaxPool�
,sequential_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_28/Conv2D/ReadVariableOp�
sequential_9/conv2d_28/Conv2DConv2D.sequential_9/max_pooling2d_27/MaxPool:output:04sequential_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_9/conv2d_28/Conv2D�
-sequential_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�
sequential_9/conv2d_28/BiasAddBiasAdd&sequential_9/conv2d_28/Conv2D:output:05sequential_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_9/conv2d_28/BiasAdd�
sequential_9/conv2d_28/ReluRelu'sequential_9/conv2d_28/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_9/conv2d_28/Relu�
%sequential_9/max_pooling2d_28/MaxPoolMaxPool)sequential_9/conv2d_28/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_28/MaxPool�
,sequential_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_29/Conv2D/ReadVariableOp�
sequential_9/conv2d_29/Conv2DConv2D.sequential_9/max_pooling2d_28/MaxPool:output:04sequential_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_9/conv2d_29/Conv2D�
-sequential_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�
sequential_9/conv2d_29/BiasAddBiasAdd&sequential_9/conv2d_29/Conv2D:output:05sequential_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_9/conv2d_29/BiasAdd�
sequential_9/conv2d_29/ReluRelu'sequential_9/conv2d_29/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_9/conv2d_29/Relu�
%sequential_9/max_pooling2d_29/MaxPoolMaxPool)sequential_9/conv2d_29/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_29/MaxPool�
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_29/MaxPool:output:0*
T0*(
_output_shapes
:�		�2"
 sequential_9/dropout_27/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*!
_output_shapes
:���2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp�
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_27/MatMul�
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp�
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_27/BiasAdd�
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_27/Relu�
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_28/Identity�
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp�
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_28/MatMul�
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp�
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_28/BiasAdd�
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_28/Relu�
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_29/Identity�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_29/BiasAddt
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_29/Softmax�
IdentityIdentitydense_29/Softmax:softmax:0 ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_27/BiasAdd/ReadVariableOp-^sequential_9/conv2d_27/Conv2D/ReadVariableOp.^sequential_9/conv2d_28/BiasAdd/ReadVariableOp-^sequential_9/conv2d_28/Conv2D/ReadVariableOp.^sequential_9/conv2d_29/BiasAdd/ReadVariableOp-^sequential_9/conv2d_29/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:�KK: : : : : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp-sequential_9/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_27/Conv2D/ReadVariableOp,sequential_9/conv2d_27/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp-sequential_9/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_28/Conv2D/ReadVariableOp,sequential_9/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp-sequential_9/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_29/Conv2D/ReadVariableOp,sequential_9/conv2d_29/Conv2D/ReadVariableOp2\
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
H
,__inference_dropout_28_layer_call_fn_1911285

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
G__inference_dropout_28_layer_call_and_return_conditional_losses_19088572
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
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1910979

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
F__inference_conv2d_29_layer_call_and_return_conditional_losses_1908807

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
�
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1911226

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
f
G__inference_dropout_27_layer_call_and_return_conditional_losses_1909031

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
�
i
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1908692

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
__inference_loss_fn_0_1911360U
;conv2d_27_kernel_regularizer_square_readvariableop_resource: 
identity��2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_27_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
IdentityIdentity$conv2d_27/kernel/Regularizer/mul:z:03^conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp
�
e
G__inference_dropout_28_layer_call_and_return_conditional_losses_1911268

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
F__inference_conv2d_28_layer_call_and_return_conditional_losses_1911164

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
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1909097

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

�
E__inference_dense_29_layer_call_and_return_conditional_losses_1909465

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
�
E__inference_dense_28_layer_call_and_return_conditional_losses_1911313

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911033

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
�
F__inference_conv2d_29_layer_call_and_return_conditional_losses_1911184

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
e
,__inference_dropout_28_layer_call_fn_1911290

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
G__inference_dropout_28_layer_call_and_return_conditional_losses_19089922
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
�
�
*__inference_dense_28_layer_call_fn_1911322

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
E__inference_dense_28_layer_call_and_return_conditional_losses_19088762
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
�
e
,__inference_dropout_29_layer_call_fn_1911349

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
G__inference_dropout_29_layer_call_and_return_conditional_losses_19089592
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
�
�
+__inference_conv2d_27_layer_call_fn_1911153

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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_19087712
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
�
@__inference_CNN_layer_call_and_return_conditional_losses_1909967

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_27_conv2d_readvariableop_resource: D
6sequential_9_conv2d_27_biasadd_readvariableop_resource: P
5sequential_9_conv2d_28_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_28_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_29_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_29_biasadd_readvariableop_resource:	�I
4sequential_9_dense_27_matmul_readvariableop_resource:���D
5sequential_9_dense_27_biasadd_readvariableop_resource:	�H
4sequential_9_dense_28_matmul_readvariableop_resource:
��D
5sequential_9_dense_28_biasadd_readvariableop_resource:	�:
'dense_29_matmul_readvariableop_resource:	�6
(dense_29_biasadd_readvariableop_resource:
identity��2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�,sequential_9/conv2d_27/Conv2D/ReadVariableOp�-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�,sequential_9/conv2d_28/Conv2D/ReadVariableOp�-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�,sequential_9/conv2d_29/Conv2D/ReadVariableOp�,sequential_9/dense_27/BiasAdd/ReadVariableOp�+sequential_9/dense_27/MatMul/ReadVariableOp�,sequential_9/dense_28/BiasAdd/ReadVariableOp�+sequential_9/dense_28/MatMul/ReadVariableOp�
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
,sequential_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_27/Conv2D/ReadVariableOp�
sequential_9/conv2d_27/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_27/Conv2D�
-sequential_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�
sequential_9/conv2d_27/BiasAddBiasAdd&sequential_9/conv2d_27/Conv2D:output:05sequential_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_27/BiasAdd�
sequential_9/conv2d_27/ReluRelu'sequential_9/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_27/Relu�
%sequential_9/max_pooling2d_27/MaxPoolMaxPool)sequential_9/conv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_27/MaxPool�
,sequential_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_28/Conv2D/ReadVariableOp�
sequential_9/conv2d_28/Conv2DConv2D.sequential_9/max_pooling2d_27/MaxPool:output:04sequential_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_28/Conv2D�
-sequential_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�
sequential_9/conv2d_28/BiasAddBiasAdd&sequential_9/conv2d_28/Conv2D:output:05sequential_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_28/BiasAdd�
sequential_9/conv2d_28/ReluRelu'sequential_9/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_28/Relu�
%sequential_9/max_pooling2d_28/MaxPoolMaxPool)sequential_9/conv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_28/MaxPool�
,sequential_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_29/Conv2D/ReadVariableOp�
sequential_9/conv2d_29/Conv2DConv2D.sequential_9/max_pooling2d_28/MaxPool:output:04sequential_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_29/Conv2D�
-sequential_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�
sequential_9/conv2d_29/BiasAddBiasAdd&sequential_9/conv2d_29/Conv2D:output:05sequential_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_29/BiasAdd�
sequential_9/conv2d_29/ReluRelu'sequential_9/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_29/Relu�
%sequential_9/max_pooling2d_29/MaxPoolMaxPool)sequential_9/conv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_29/MaxPool�
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_29/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_9/dropout_27/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp�
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/MatMul�
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp�
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/BiasAdd�
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/Relu�
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_28/Identity�
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp�
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/MatMul�
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp�
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/BiasAdd�
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/Relu�
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_29/Identity�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_29/Softmax�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentitydense_29/Softmax:softmax:03^conv2d_27/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_27/BiasAdd/ReadVariableOp-^sequential_9/conv2d_27/Conv2D/ReadVariableOp.^sequential_9/conv2d_28/BiasAdd/ReadVariableOp-^sequential_9/conv2d_28/Conv2D/ReadVariableOp.^sequential_9/conv2d_29/BiasAdd/ReadVariableOp-^sequential_9/conv2d_29/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp-sequential_9/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_27/Conv2D/ReadVariableOp,sequential_9/conv2d_27/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp-sequential_9/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_28/Conv2D/ReadVariableOp,sequential_9/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp-sequential_9/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_29/Conv2D/ReadVariableOp,sequential_9/conv2d_29/Conv2D/ReadVariableOp2\
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_9_layer_call_fn_1911095

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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_19086142
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
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1909124

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
G__inference_dropout_27_layer_call_and_return_conditional_losses_1908819

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
%__inference_CNN_layer_call_fn_1910316
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
@__inference_CNN_layer_call_and_return_conditional_losses_19094902
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
�
@__inference_CNN_layer_call_and_return_conditional_losses_1910078

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_27_conv2d_readvariableop_resource: D
6sequential_9_conv2d_27_biasadd_readvariableop_resource: P
5sequential_9_conv2d_28_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_28_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_29_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_29_biasadd_readvariableop_resource:	�I
4sequential_9_dense_27_matmul_readvariableop_resource:���D
5sequential_9_dense_27_biasadd_readvariableop_resource:	�H
4sequential_9_dense_28_matmul_readvariableop_resource:
��D
5sequential_9_dense_28_biasadd_readvariableop_resource:	�:
'dense_29_matmul_readvariableop_resource:	�6
(dense_29_biasadd_readvariableop_resource:
identity��2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�1sequential_9/batch_normalization_9/AssignNewValue�3sequential_9/batch_normalization_9/AssignNewValue_1�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�,sequential_9/conv2d_27/Conv2D/ReadVariableOp�-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�,sequential_9/conv2d_28/Conv2D/ReadVariableOp�-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�,sequential_9/conv2d_29/Conv2D/ReadVariableOp�,sequential_9/dense_27/BiasAdd/ReadVariableOp�+sequential_9/dense_27/MatMul/ReadVariableOp�,sequential_9/dense_28/BiasAdd/ReadVariableOp�+sequential_9/dense_28/MatMul/ReadVariableOp�
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
,sequential_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_27/Conv2D/ReadVariableOp�
sequential_9/conv2d_27/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_27/Conv2D�
-sequential_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�
sequential_9/conv2d_27/BiasAddBiasAdd&sequential_9/conv2d_27/Conv2D:output:05sequential_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_27/BiasAdd�
sequential_9/conv2d_27/ReluRelu'sequential_9/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_27/Relu�
%sequential_9/max_pooling2d_27/MaxPoolMaxPool)sequential_9/conv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_27/MaxPool�
,sequential_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_28/Conv2D/ReadVariableOp�
sequential_9/conv2d_28/Conv2DConv2D.sequential_9/max_pooling2d_27/MaxPool:output:04sequential_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_28/Conv2D�
-sequential_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�
sequential_9/conv2d_28/BiasAddBiasAdd&sequential_9/conv2d_28/Conv2D:output:05sequential_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_28/BiasAdd�
sequential_9/conv2d_28/ReluRelu'sequential_9/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_28/Relu�
%sequential_9/max_pooling2d_28/MaxPoolMaxPool)sequential_9/conv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_28/MaxPool�
,sequential_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_29/Conv2D/ReadVariableOp�
sequential_9/conv2d_29/Conv2DConv2D.sequential_9/max_pooling2d_28/MaxPool:output:04sequential_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_29/Conv2D�
-sequential_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�
sequential_9/conv2d_29/BiasAddBiasAdd&sequential_9/conv2d_29/Conv2D:output:05sequential_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_29/BiasAdd�
sequential_9/conv2d_29/ReluRelu'sequential_9/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_29/Relu�
%sequential_9/max_pooling2d_29/MaxPoolMaxPool)sequential_9/conv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_29/MaxPool�
%sequential_9/dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2'
%sequential_9/dropout_27/dropout/Const�
#sequential_9/dropout_27/dropout/MulMul.sequential_9/max_pooling2d_29/MaxPool:output:0.sequential_9/dropout_27/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2%
#sequential_9/dropout_27/dropout/Mul�
%sequential_9/dropout_27/dropout/ShapeShape.sequential_9/max_pooling2d_29/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_27/dropout/Shape�
<sequential_9/dropout_27/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_27/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype02>
<sequential_9/dropout_27/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=20
.sequential_9/dropout_27/dropout/GreaterEqual/y�
,sequential_9/dropout_27/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_27/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_27/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2.
,sequential_9/dropout_27/dropout/GreaterEqual�
$sequential_9/dropout_27/dropout/CastCast0sequential_9/dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2&
$sequential_9/dropout_27/dropout/Cast�
%sequential_9/dropout_27/dropout/Mul_1Mul'sequential_9/dropout_27/dropout/Mul:z:0(sequential_9/dropout_27/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2'
%sequential_9/dropout_27/dropout/Mul_1�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/dropout/Mul_1:z:0%sequential_9/flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp�
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/MatMul�
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp�
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/BiasAdd�
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/Relu�
%sequential_9/dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_28/dropout/Const�
#sequential_9/dropout_28/dropout/MulMul(sequential_9/dense_27/Relu:activations:0.sequential_9/dropout_28/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_28/dropout/Mul�
%sequential_9/dropout_28/dropout/ShapeShape(sequential_9/dense_27/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_28/dropout/Shape�
<sequential_9/dropout_28/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_28/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_28/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_28/dropout/GreaterEqual/y�
,sequential_9/dropout_28/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_28/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_28/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_28/dropout/GreaterEqual�
$sequential_9/dropout_28/dropout/CastCast0sequential_9/dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_28/dropout/Cast�
%sequential_9/dropout_28/dropout/Mul_1Mul'sequential_9/dropout_28/dropout/Mul:z:0(sequential_9/dropout_28/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_28/dropout/Mul_1�
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp�
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/dropout/Mul_1:z:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/MatMul�
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp�
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/BiasAdd�
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/Relu�
%sequential_9/dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_29/dropout/Const�
#sequential_9/dropout_29/dropout/MulMul(sequential_9/dense_28/Relu:activations:0.sequential_9/dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_29/dropout/Mul�
%sequential_9/dropout_29/dropout/ShapeShape(sequential_9/dense_28/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_29/dropout/Shape�
<sequential_9/dropout_29/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_29/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_29/dropout/GreaterEqual/y�
,sequential_9/dropout_29/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_29/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_29/dropout/GreaterEqual�
$sequential_9/dropout_29/dropout/CastCast0sequential_9/dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_29/dropout/Cast�
%sequential_9/dropout_29/dropout/Mul_1Mul'sequential_9/dropout_29/dropout/Mul:z:0(sequential_9/dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_29/dropout/Mul_1�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMul)sequential_9/dropout_29/dropout/Mul_1:z:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_29/Softmax�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�	
IdentityIdentitydense_29/Softmax:softmax:03^conv2d_27/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_27/BiasAdd/ReadVariableOp-^sequential_9/conv2d_27/Conv2D/ReadVariableOp.^sequential_9/conv2d_28/BiasAdd/ReadVariableOp-^sequential_9/conv2d_28/Conv2D/ReadVariableOp.^sequential_9/conv2d_29/BiasAdd/ReadVariableOp-^sequential_9/conv2d_29/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_12�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp-sequential_9/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_27/Conv2D/ReadVariableOp,sequential_9/conv2d_27/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp-sequential_9/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_28/Conv2D/ReadVariableOp,sequential_9/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp-sequential_9/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_29/Conv2D/ReadVariableOp,sequential_9/conv2d_29/Conv2D/ReadVariableOp2\
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
.__inference_sequential_9_layer_call_fn_1910852
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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_19089082
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
_user_specified_namelambda_9_input
�
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1908725

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
Қ
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910819
lambda_9_input;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_27_conv2d_readvariableop_resource: 7
)conv2d_27_biasadd_readvariableop_resource: C
(conv2d_28_conv2d_readvariableop_resource: �8
)conv2d_28_biasadd_readvariableop_resource:	�D
(conv2d_29_conv2d_readvariableop_resource:��8
)conv2d_29_biasadd_readvariableop_resource:	�<
'dense_27_matmul_readvariableop_resource:���7
(dense_27_biasadd_readvariableop_resource:	�;
'dense_28_matmul_readvariableop_resource:
��7
(dense_28_biasadd_readvariableop_resource:	�
identity��$batch_normalization_9/AssignNewValue�&batch_normalization_9/AssignNewValue_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp�2conv2d_27/kernel/Regularizer/Square/ReadVariableOp� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�
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
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_27/Relu�
max_pooling2d_27/MaxPoolMaxPoolconv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2DConv2D!max_pooling2d_27/MaxPool:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_28/Conv2D�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_28/BiasAdd
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_28/Relu�
max_pooling2d_28/MaxPoolMaxPoolconv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_28/MaxPool�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_29/Conv2D�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_29/BiasAdd
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_29/Relu�
max_pooling2d_29/MaxPoolMaxPoolconv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_29/MaxPooly
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_27/dropout/Const�
dropout_27/dropout/MulMul!max_pooling2d_29/MaxPool:output:0!dropout_27/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2
dropout_27/dropout/Mul�
dropout_27/dropout/ShapeShape!max_pooling2d_29/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shape�
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype021
/dropout_27/dropout/random_uniform/RandomUniform�
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_27/dropout/GreaterEqual/y�
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2!
dropout_27/dropout/GreaterEqual�
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout_27/dropout/Cast�
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout_27/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_27/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_9/Reshape�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_27/Reluy
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_28/dropout/Const�
dropout_28/dropout/MulMuldense_27/Relu:activations:0!dropout_28/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_28/dropout/Mul
dropout_28/dropout/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
dropout_28/dropout/Shape�
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_28/dropout/random_uniform/RandomUniform�
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_28/dropout/GreaterEqual/y�
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_28/dropout/GreaterEqual�
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_28/dropout/Cast�
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_28/dropout/Mul_1�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_28/MatMul/ReadVariableOp�
dense_28/MatMulMatMuldropout_28/dropout/Mul_1:z:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_28/MatMul�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_28/BiasAdd/ReadVariableOp�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_28/Reluy
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_29/dropout/Const�
dropout_29/dropout/MulMuldense_28/Relu:activations:0!dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_29/dropout/Mul
dropout_29/dropout/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:2
dropout_29/dropout/Shape�
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_29/dropout/random_uniform/RandomUniform�
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_29/dropout/GreaterEqual/y�
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_29/dropout/GreaterEqual�
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_29/dropout/Cast�
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_29/dropout/Mul_1�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentitydropout_29/dropout/Mul_1:z:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_9_input
�
f
G__inference_dropout_27_layer_call_and_return_conditional_losses_1911210

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
�
e
G__inference_dropout_27_layer_call_and_return_conditional_losses_1911198

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
�v
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910715
lambda_9_input;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_27_conv2d_readvariableop_resource: 7
)conv2d_27_biasadd_readvariableop_resource: C
(conv2d_28_conv2d_readvariableop_resource: �8
)conv2d_28_biasadd_readvariableop_resource:	�D
(conv2d_29_conv2d_readvariableop_resource:��8
)conv2d_29_biasadd_readvariableop_resource:	�<
'dense_27_matmul_readvariableop_resource:���7
(dense_27_biasadd_readvariableop_resource:	�;
'dense_28_matmul_readvariableop_resource:
��7
(dense_28_biasadd_readvariableop_resource:	�
identity��5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp�2conv2d_27/kernel/Regularizer/Square/ReadVariableOp� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�
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
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_27/Relu�
max_pooling2d_27/MaxPoolMaxPoolconv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2DConv2D!max_pooling2d_27/MaxPool:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_28/Conv2D�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_28/BiasAdd
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_28/Relu�
max_pooling2d_28/MaxPoolMaxPoolconv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_28/MaxPool�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_29/Conv2D�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_29/BiasAdd
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_29/Relu�
max_pooling2d_29/MaxPoolMaxPoolconv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_29/MaxPool�
dropout_27/IdentityIdentity!max_pooling2d_29/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2
dropout_27/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_27/Identity:output:0flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_9/Reshape�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_27/Relu�
dropout_28/IdentityIdentitydense_27/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_28/Identity�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_28/MatMul/ReadVariableOp�
dense_28/MatMulMatMuldropout_28/Identity:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_28/MatMul�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_28/BiasAdd/ReadVariableOp�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_28/Relu�
dropout_29/IdentityIdentitydense_28/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_29/Identity�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentitydropout_29/Identity:output:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:���������KK
(
_user_specified_namelambda_9_input
�
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1908827

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
�
i
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1908704

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
�
.__inference_sequential_9_layer_call_fn_1910918

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
I__inference_sequential_9_layer_call_and_return_conditional_losses_19092262
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
�
�
E__inference_dense_28_layer_call_and_return_conditional_losses_1908876

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_9_layer_call_fn_1910885

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
I__inference_sequential_9_layer_call_and_return_conditional_losses_19089082
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
�u
�
__inference_call_1790911

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_27_conv2d_readvariableop_resource: D
6sequential_9_conv2d_27_biasadd_readvariableop_resource: P
5sequential_9_conv2d_28_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_28_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_29_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_29_biasadd_readvariableop_resource:	�I
4sequential_9_dense_27_matmul_readvariableop_resource:���D
5sequential_9_dense_27_biasadd_readvariableop_resource:	�H
4sequential_9_dense_28_matmul_readvariableop_resource:
��D
5sequential_9_dense_28_biasadd_readvariableop_resource:	�:
'dense_29_matmul_readvariableop_resource:	�6
(dense_29_biasadd_readvariableop_resource:
identity��dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�,sequential_9/conv2d_27/Conv2D/ReadVariableOp�-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�,sequential_9/conv2d_28/Conv2D/ReadVariableOp�-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�,sequential_9/conv2d_29/Conv2D/ReadVariableOp�,sequential_9/dense_27/BiasAdd/ReadVariableOp�+sequential_9/dense_27/MatMul/ReadVariableOp�,sequential_9/dense_28/BiasAdd/ReadVariableOp�+sequential_9/dense_28/MatMul/ReadVariableOp�
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
,sequential_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_27/Conv2D/ReadVariableOp�
sequential_9/conv2d_27/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_27/Conv2D�
-sequential_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�
sequential_9/conv2d_27/BiasAddBiasAdd&sequential_9/conv2d_27/Conv2D:output:05sequential_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_27/BiasAdd�
sequential_9/conv2d_27/ReluRelu'sequential_9/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_27/Relu�
%sequential_9/max_pooling2d_27/MaxPoolMaxPool)sequential_9/conv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_27/MaxPool�
,sequential_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_28/Conv2D/ReadVariableOp�
sequential_9/conv2d_28/Conv2DConv2D.sequential_9/max_pooling2d_27/MaxPool:output:04sequential_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_28/Conv2D�
-sequential_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�
sequential_9/conv2d_28/BiasAddBiasAdd&sequential_9/conv2d_28/Conv2D:output:05sequential_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_28/BiasAdd�
sequential_9/conv2d_28/ReluRelu'sequential_9/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_28/Relu�
%sequential_9/max_pooling2d_28/MaxPoolMaxPool)sequential_9/conv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_28/MaxPool�
,sequential_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_29/Conv2D/ReadVariableOp�
sequential_9/conv2d_29/Conv2DConv2D.sequential_9/max_pooling2d_28/MaxPool:output:04sequential_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_29/Conv2D�
-sequential_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�
sequential_9/conv2d_29/BiasAddBiasAdd&sequential_9/conv2d_29/Conv2D:output:05sequential_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_29/BiasAdd�
sequential_9/conv2d_29/ReluRelu'sequential_9/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_29/Relu�
%sequential_9/max_pooling2d_29/MaxPoolMaxPool)sequential_9/conv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_29/MaxPool�
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_29/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_9/dropout_27/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp�
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/MatMul�
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp�
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/BiasAdd�
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/Relu�
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_28/Identity�
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp�
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/MatMul�
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp�
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/BiasAdd�
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/Relu�
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_29/Identity�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_29/Softmax�
IdentityIdentitydense_29/Softmax:softmax:0 ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_27/BiasAdd/ReadVariableOp-^sequential_9/conv2d_27/Conv2D/ReadVariableOp.^sequential_9/conv2d_28/BiasAdd/ReadVariableOp-^sequential_9/conv2d_28/Conv2D/ReadVariableOp.^sequential_9/conv2d_29/BiasAdd/ReadVariableOp-^sequential_9/conv2d_29/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp-sequential_9/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_27/Conv2D/ReadVariableOp,sequential_9/conv2d_27/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp-sequential_9/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_28/Conv2D/ReadVariableOp,sequential_9/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp-sequential_9/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_29/Conv2D/ReadVariableOp,sequential_9/conv2d_29/Conv2D/ReadVariableOp2\
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
+__inference_conv2d_28_layer_call_fn_1911173

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
F__inference_conv2d_28_layer_call_and_return_conditional_losses_19087892
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
�

�
E__inference_dense_29_layer_call_and_return_conditional_losses_1910962

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
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1908570

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
.__inference_sequential_9_layer_call_fn_1910951
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
	unknown_8:	�
	unknown_9:���

unknown_10:	�

unknown_11:
��

unknown_12:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_19092262
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
_user_specified_namelambda_9_input
��
�!
#__inference__traced_restore_1911733
file_prefix3
 assignvariableop_dense_29_kernel:	�.
 assignvariableop_1_dense_29_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_9_gamma:;
-assignvariableop_8_batch_normalization_9_beta:=
#assignvariableop_9_conv2d_27_kernel: 0
"assignvariableop_10_conv2d_27_bias: ?
$assignvariableop_11_conv2d_28_kernel: �1
"assignvariableop_12_conv2d_28_bias:	�@
$assignvariableop_13_conv2d_29_kernel:��1
"assignvariableop_14_conv2d_29_bias:	�8
#assignvariableop_15_dense_27_kernel:���0
!assignvariableop_16_dense_27_bias:	�7
#assignvariableop_17_dense_28_kernel:
��0
!assignvariableop_18_dense_28_bias:	�C
5assignvariableop_19_batch_normalization_9_moving_mean:G
9assignvariableop_20_batch_normalization_9_moving_variance:#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_29_kernel_m:	�6
(assignvariableop_26_adam_dense_29_bias_m:D
6assignvariableop_27_adam_batch_normalization_9_gamma_m:C
5assignvariableop_28_adam_batch_normalization_9_beta_m:E
+assignvariableop_29_adam_conv2d_27_kernel_m: 7
)assignvariableop_30_adam_conv2d_27_bias_m: F
+assignvariableop_31_adam_conv2d_28_kernel_m: �8
)assignvariableop_32_adam_conv2d_28_bias_m:	�G
+assignvariableop_33_adam_conv2d_29_kernel_m:��8
)assignvariableop_34_adam_conv2d_29_bias_m:	�?
*assignvariableop_35_adam_dense_27_kernel_m:���7
(assignvariableop_36_adam_dense_27_bias_m:	�>
*assignvariableop_37_adam_dense_28_kernel_m:
��7
(assignvariableop_38_adam_dense_28_bias_m:	�=
*assignvariableop_39_adam_dense_29_kernel_v:	�6
(assignvariableop_40_adam_dense_29_bias_v:D
6assignvariableop_41_adam_batch_normalization_9_gamma_v:C
5assignvariableop_42_adam_batch_normalization_9_beta_v:E
+assignvariableop_43_adam_conv2d_27_kernel_v: 7
)assignvariableop_44_adam_conv2d_27_bias_v: F
+assignvariableop_45_adam_conv2d_28_kernel_v: �8
)assignvariableop_46_adam_conv2d_28_bias_v:	�G
+assignvariableop_47_adam_conv2d_29_kernel_v:��8
)assignvariableop_48_adam_conv2d_29_bias_v:	�?
*assignvariableop_49_adam_dense_27_kernel_v:���7
(assignvariableop_50_adam_dense_27_bias_v:	�>
*assignvariableop_51_adam_dense_28_kernel_v:
��7
(assignvariableop_52_adam_dense_28_bias_v:	�
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
AssignVariableOpAssignVariableOp assignvariableop_dense_29_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_29_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_27_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_27_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_28_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_28_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_29_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_29_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_27_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_27_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_28_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_28_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_9_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_9_moving_varianceIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_29_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_29_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_9_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_9_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_27_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_27_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_28_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_28_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_29_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_29_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_27_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_27_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_28_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_28_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_29_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_29_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_9_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_9_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_27_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_27_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_28_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_28_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_29_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_29_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_27_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_27_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_28_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_28_bias_vIdentity_52:output:0"/device:CPU:0*
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
�
�
%__inference_CNN_layer_call_fn_1910390

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
@__inference_CNN_layer_call_and_return_conditional_losses_19096302
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
�
F
*__inference_lambda_9_layer_call_fn_1910992

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
E__inference_lambda_9_layer_call_and_return_conditional_losses_19087252
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
�
N
2__inference_max_pooling2d_29_layer_call_fn_1908710

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
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_19087042
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911015

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
f
G__inference_dropout_28_layer_call_and_return_conditional_losses_1911280

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
__inference_loss_fn_2_1911382N
:dense_28_kernel_regularizer_square_readvariableop_resource:
��
identity��1dense_28/kernel/Regularizer/Square/ReadVariableOp�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_28_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentity#dense_28/kernel/Regularizer/mul:z:02^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp
�
�
%__inference_CNN_layer_call_fn_1910353

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
@__inference_CNN_layer_call_and_return_conditional_losses_19094902
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
�\
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1908908

inputs+
batch_normalization_9_1908745:+
batch_normalization_9_1908747:+
batch_normalization_9_1908749:+
batch_normalization_9_1908751:+
conv2d_27_1908772: 
conv2d_27_1908774: ,
conv2d_28_1908790: � 
conv2d_28_1908792:	�-
conv2d_29_1908808:�� 
conv2d_29_1908810:	�%
dense_27_1908847:���
dense_27_1908849:	�$
dense_28_1908877:
��
dense_28_1908879:	�
identity��-batch_normalization_9/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�1dense_27/kernel/Regularizer/Square/ReadVariableOp� dense_28/StatefulPartitionedCall�1dense_28/kernel/Regularizer/Square/ReadVariableOp�
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
E__inference_lambda_9_layer_call_and_return_conditional_losses_19087252
lambda_9/PartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall!lambda_9/PartitionedCall:output:0batch_normalization_9_1908745batch_normalization_9_1908747batch_normalization_9_1908749batch_normalization_9_1908751*
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_19087442/
-batch_normalization_9/StatefulPartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_27_1908772conv2d_27_1908774*
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_19087712#
!conv2d_27/StatefulPartitionedCall�
 max_pooling2d_27/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_19086802"
 max_pooling2d_27/PartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_27/PartitionedCall:output:0conv2d_28_1908790conv2d_28_1908792*
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
F__inference_conv2d_28_layer_call_and_return_conditional_losses_19087892#
!conv2d_28/StatefulPartitionedCall�
 max_pooling2d_28/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_19086922"
 max_pooling2d_28/PartitionedCall�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_28/PartitionedCall:output:0conv2d_29_1908808conv2d_29_1908810*
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
F__inference_conv2d_29_layer_call_and_return_conditional_losses_19088072#
!conv2d_29/StatefulPartitionedCall�
 max_pooling2d_29/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_19087042"
 max_pooling2d_29/PartitionedCall�
dropout_27/PartitionedCallPartitionedCall)max_pooling2d_29/PartitionedCall:output:0*
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
G__inference_dropout_27_layer_call_and_return_conditional_losses_19088192
dropout_27/PartitionedCall�
flatten_9/PartitionedCallPartitionedCall#dropout_27/PartitionedCall:output:0*
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
F__inference_flatten_9_layer_call_and_return_conditional_losses_19088272
flatten_9/PartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_27_1908847dense_27_1908849*
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
E__inference_dense_27_layer_call_and_return_conditional_losses_19088462"
 dense_27/StatefulPartitionedCall�
dropout_28/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
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
G__inference_dropout_28_layer_call_and_return_conditional_losses_19088572
dropout_28/PartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0dense_28_1908877dense_28_1908879*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_19088762"
 dense_28/StatefulPartitionedCall�
dropout_29/PartitionedCallPartitionedCall)dense_28/StatefulPartitionedCall:output:0*
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
G__inference_dropout_29_layer_call_and_return_conditional_losses_19088872
dropout_29/PartitionedCall�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_1908772*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_27_1908847*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_28_1908877* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentity#dropout_29/PartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall2^dense_27/kernel/Regularizer/Square/ReadVariableOp!^dense_28/StatefulPartitionedCall2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�u
�
__inference_call_1788781

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_27_conv2d_readvariableop_resource: D
6sequential_9_conv2d_27_biasadd_readvariableop_resource: P
5sequential_9_conv2d_28_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_28_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_29_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_29_biasadd_readvariableop_resource:	�I
4sequential_9_dense_27_matmul_readvariableop_resource:���D
5sequential_9_dense_27_biasadd_readvariableop_resource:	�H
4sequential_9_dense_28_matmul_readvariableop_resource:
��D
5sequential_9_dense_28_biasadd_readvariableop_resource:	�:
'dense_29_matmul_readvariableop_resource:	�6
(dense_29_biasadd_readvariableop_resource:
identity��dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�,sequential_9/conv2d_27/Conv2D/ReadVariableOp�-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�,sequential_9/conv2d_28/Conv2D/ReadVariableOp�-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�,sequential_9/conv2d_29/Conv2D/ReadVariableOp�,sequential_9/dense_27/BiasAdd/ReadVariableOp�+sequential_9/dense_27/MatMul/ReadVariableOp�,sequential_9/dense_28/BiasAdd/ReadVariableOp�+sequential_9/dense_28/MatMul/ReadVariableOp�
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
,sequential_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_27/Conv2D/ReadVariableOp�
sequential_9/conv2d_27/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_27/Conv2D�
-sequential_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�
sequential_9/conv2d_27/BiasAddBiasAdd&sequential_9/conv2d_27/Conv2D:output:05sequential_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_27/BiasAdd�
sequential_9/conv2d_27/ReluRelu'sequential_9/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_27/Relu�
%sequential_9/max_pooling2d_27/MaxPoolMaxPool)sequential_9/conv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_27/MaxPool�
,sequential_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_28/Conv2D/ReadVariableOp�
sequential_9/conv2d_28/Conv2DConv2D.sequential_9/max_pooling2d_27/MaxPool:output:04sequential_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_28/Conv2D�
-sequential_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�
sequential_9/conv2d_28/BiasAddBiasAdd&sequential_9/conv2d_28/Conv2D:output:05sequential_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_28/BiasAdd�
sequential_9/conv2d_28/ReluRelu'sequential_9/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_28/Relu�
%sequential_9/max_pooling2d_28/MaxPoolMaxPool)sequential_9/conv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_28/MaxPool�
,sequential_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_29/Conv2D/ReadVariableOp�
sequential_9/conv2d_29/Conv2DConv2D.sequential_9/max_pooling2d_28/MaxPool:output:04sequential_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_29/Conv2D�
-sequential_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�
sequential_9/conv2d_29/BiasAddBiasAdd&sequential_9/conv2d_29/Conv2D:output:05sequential_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_29/BiasAdd�
sequential_9/conv2d_29/ReluRelu'sequential_9/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_29/Relu�
%sequential_9/max_pooling2d_29/MaxPoolMaxPool)sequential_9/conv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_29/MaxPool�
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_29/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2"
 sequential_9/dropout_27/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp�
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/MatMul�
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp�
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/BiasAdd�
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/Relu�
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_28/Identity�
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp�
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/MatMul�
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp�
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/BiasAdd�
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/Relu�
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0*(
_output_shapes
:����������2"
 sequential_9/dropout_29/Identity�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_29/Softmax�
IdentityIdentitydense_29/Softmax:softmax:0 ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_27/BiasAdd/ReadVariableOp-^sequential_9/conv2d_27/Conv2D/ReadVariableOp.^sequential_9/conv2d_28/BiasAdd/ReadVariableOp-^sequential_9/conv2d_28/Conv2D/ReadVariableOp.^sequential_9/conv2d_29/BiasAdd/ReadVariableOp-^sequential_9/conv2d_29/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp-sequential_9/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_27/Conv2D/ReadVariableOp,sequential_9/conv2d_27/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp-sequential_9/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_28/Conv2D/ReadVariableOp,sequential_9/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp-sequential_9/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_29/Conv2D/ReadVariableOp,sequential_9/conv2d_29/Conv2D/ReadVariableOp2\
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
e
G__inference_dropout_28_layer_call_and_return_conditional_losses_1908857

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
G__inference_dropout_29_layer_call_and_return_conditional_losses_1908959

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
F
*__inference_lambda_9_layer_call_fn_1910997

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
E__inference_lambda_9_layer_call_and_return_conditional_losses_19091242
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
H
,__inference_dropout_29_layer_call_fn_1911344

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
G__inference_dropout_29_layer_call_and_return_conditional_losses_19088872
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911051

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
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1908744

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
�s
�
__inference_call_1790767

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_27_conv2d_readvariableop_resource: D
6sequential_9_conv2d_27_biasadd_readvariableop_resource: P
5sequential_9_conv2d_28_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_28_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_29_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_29_biasadd_readvariableop_resource:	�I
4sequential_9_dense_27_matmul_readvariableop_resource:���D
5sequential_9_dense_27_biasadd_readvariableop_resource:	�H
4sequential_9_dense_28_matmul_readvariableop_resource:
��D
5sequential_9_dense_28_biasadd_readvariableop_resource:	�:
'dense_29_matmul_readvariableop_resource:	�6
(dense_29_biasadd_readvariableop_resource:
identity��dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�,sequential_9/conv2d_27/Conv2D/ReadVariableOp�-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�,sequential_9/conv2d_28/Conv2D/ReadVariableOp�-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�,sequential_9/conv2d_29/Conv2D/ReadVariableOp�,sequential_9/dense_27/BiasAdd/ReadVariableOp�+sequential_9/dense_27/MatMul/ReadVariableOp�,sequential_9/dense_28/BiasAdd/ReadVariableOp�+sequential_9/dense_28/MatMul/ReadVariableOp�
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
,sequential_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_27/Conv2D/ReadVariableOp�
sequential_9/conv2d_27/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK *
paddingSAME*
strides
2
sequential_9/conv2d_27/Conv2D�
-sequential_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�
sequential_9/conv2d_27/BiasAddBiasAdd&sequential_9/conv2d_27/Conv2D:output:05sequential_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�KK 2 
sequential_9/conv2d_27/BiasAdd�
sequential_9/conv2d_27/ReluRelu'sequential_9/conv2d_27/BiasAdd:output:0*
T0*'
_output_shapes
:�KK 2
sequential_9/conv2d_27/Relu�
%sequential_9/max_pooling2d_27/MaxPoolMaxPool)sequential_9/conv2d_27/Relu:activations:0*'
_output_shapes
:�%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_27/MaxPool�
,sequential_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_28/Conv2D/ReadVariableOp�
sequential_9/conv2d_28/Conv2DConv2D.sequential_9/max_pooling2d_27/MaxPool:output:04sequential_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�*
paddingSAME*
strides
2
sequential_9/conv2d_28/Conv2D�
-sequential_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�
sequential_9/conv2d_28/BiasAddBiasAdd&sequential_9/conv2d_28/Conv2D:output:05sequential_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�%%�2 
sequential_9/conv2d_28/BiasAdd�
sequential_9/conv2d_28/ReluRelu'sequential_9/conv2d_28/BiasAdd:output:0*
T0*(
_output_shapes
:�%%�2
sequential_9/conv2d_28/Relu�
%sequential_9/max_pooling2d_28/MaxPoolMaxPool)sequential_9/conv2d_28/Relu:activations:0*(
_output_shapes
:��*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_28/MaxPool�
,sequential_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_29/Conv2D/ReadVariableOp�
sequential_9/conv2d_29/Conv2DConv2D.sequential_9/max_pooling2d_28/MaxPool:output:04sequential_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
2
sequential_9/conv2d_29/Conv2D�
-sequential_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�
sequential_9/conv2d_29/BiasAddBiasAdd&sequential_9/conv2d_29/Conv2D:output:05sequential_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:��2 
sequential_9/conv2d_29/BiasAdd�
sequential_9/conv2d_29/ReluRelu'sequential_9/conv2d_29/BiasAdd:output:0*
T0*(
_output_shapes
:��2
sequential_9/conv2d_29/Relu�
%sequential_9/max_pooling2d_29/MaxPoolMaxPool)sequential_9/conv2d_29/Relu:activations:0*(
_output_shapes
:�		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_29/MaxPool�
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_29/MaxPool:output:0*
T0*(
_output_shapes
:�		�2"
 sequential_9/dropout_27/Identity�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*!
_output_shapes
:���2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp�
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_27/MatMul�
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp�
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_27/BiasAdd�
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_27/Relu�
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_28/Identity�
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp�
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_28/MatMul�
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp�
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_28/BiasAdd�
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0* 
_output_shapes
:
��2
sequential_9/dense_28/Relu�
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0* 
_output_shapes
:
��2"
 sequential_9/dropout_29/Identity�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense_29/BiasAddt
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*
_output_shapes
:	�2
dense_29/Softmax�
IdentityIdentitydense_29/Softmax:softmax:0 ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_27/BiasAdd/ReadVariableOp-^sequential_9/conv2d_27/Conv2D/ReadVariableOp.^sequential_9/conv2d_28/BiasAdd/ReadVariableOp-^sequential_9/conv2d_28/Conv2D/ReadVariableOp.^sequential_9/conv2d_29/BiasAdd/ReadVariableOp-^sequential_9/conv2d_29/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:�KK: : : : : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp-sequential_9/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_27/Conv2D/ReadVariableOp,sequential_9/conv2d_27/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp-sequential_9/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_28/Conv2D/ReadVariableOp,sequential_9/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp-sequential_9/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_29/Conv2D/ReadVariableOp,sequential_9/conv2d_29/Conv2D/ReadVariableOp2\
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:O K
'
_output_shapes
:�KK
 
_user_specified_nameinputs
�
G
+__inference_flatten_9_layer_call_fn_1911231

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
F__inference_flatten_9_layer_call_and_return_conditional_losses_19088272
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
�
N
2__inference_max_pooling2d_28_layer_call_fn_1908698

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
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_19086922
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
��
�
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910632

inputs;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_27_conv2d_readvariableop_resource: 7
)conv2d_27_biasadd_readvariableop_resource: C
(conv2d_28_conv2d_readvariableop_resource: �8
)conv2d_28_biasadd_readvariableop_resource:	�D
(conv2d_29_conv2d_readvariableop_resource:��8
)conv2d_29_biasadd_readvariableop_resource:	�<
'dense_27_matmul_readvariableop_resource:���7
(dense_27_biasadd_readvariableop_resource:	�;
'dense_28_matmul_readvariableop_resource:
��7
(dense_28_biasadd_readvariableop_resource:	�
identity��$batch_normalization_9/AssignNewValue�&batch_normalization_9/AssignNewValue_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp�2conv2d_27/kernel/Regularizer/Square/ReadVariableOp� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�
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
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_27/Relu�
max_pooling2d_27/MaxPoolMaxPoolconv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2DConv2D!max_pooling2d_27/MaxPool:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_28/Conv2D�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_28/BiasAdd
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_28/Relu�
max_pooling2d_28/MaxPoolMaxPoolconv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_28/MaxPool�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_29/Conv2D�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_29/BiasAdd
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_29/Relu�
max_pooling2d_29/MaxPoolMaxPoolconv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_29/MaxPooly
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_27/dropout/Const�
dropout_27/dropout/MulMul!max_pooling2d_29/MaxPool:output:0!dropout_27/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2
dropout_27/dropout/Mul�
dropout_27/dropout/ShapeShape!max_pooling2d_29/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shape�
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype021
/dropout_27/dropout/random_uniform/RandomUniform�
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_27/dropout/GreaterEqual/y�
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2!
dropout_27/dropout/GreaterEqual�
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2
dropout_27/dropout/Cast�
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2
dropout_27/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_27/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_9/Reshape�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_27/Reluy
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_28/dropout/Const�
dropout_28/dropout/MulMuldense_27/Relu:activations:0!dropout_28/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_28/dropout/Mul
dropout_28/dropout/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
dropout_28/dropout/Shape�
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_28/dropout/random_uniform/RandomUniform�
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_28/dropout/GreaterEqual/y�
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_28/dropout/GreaterEqual�
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_28/dropout/Cast�
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_28/dropout/Mul_1�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_28/MatMul/ReadVariableOp�
dense_28/MatMulMatMuldropout_28/dropout/Mul_1:z:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_28/MatMul�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_28/BiasAdd/ReadVariableOp�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_28/Reluy
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_29/dropout/Const�
dropout_29/dropout/MulMuldense_28/Relu:activations:0!dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_29/dropout/Mul
dropout_29/dropout/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:2
dropout_29/dropout/Shape�
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_29/dropout/random_uniform/RandomUniform�
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_29/dropout/GreaterEqual/y�
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_29/dropout/GreaterEqual�
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_29/dropout/Cast�
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_29/dropout/Mul_1�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentitydropout_29/dropout/Mul_1:z:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
e
,__inference_dropout_27_layer_call_fn_1911220

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
G__inference_dropout_27_layer_call_and_return_conditional_losses_19090312
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
�
�
__inference_loss_fn_1_1911371O
:dense_27_kernel_regularizer_square_readvariableop_resource:���
identity��1dense_27/kernel/Regularizer/Square/ReadVariableOp�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_27_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
IdentityIdentity#dense_27/kernel/Regularizer/mul:z:02^dense_27/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp
�
�
F__inference_conv2d_27_layer_call_and_return_conditional_losses_1911144

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
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
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_1908680

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
�2
�
@__inference_CNN_layer_call_and_return_conditional_losses_1909490

inputs"
sequential_9_1909425:"
sequential_9_1909427:"
sequential_9_1909429:"
sequential_9_1909431:.
sequential_9_1909433: "
sequential_9_1909435: /
sequential_9_1909437: �#
sequential_9_1909439:	�0
sequential_9_1909441:��#
sequential_9_1909443:	�)
sequential_9_1909445:���#
sequential_9_1909447:	�(
sequential_9_1909449:
��#
sequential_9_1909451:	�#
dense_29_1909466:	�
dense_29_1909468:
identity��2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp� dense_29/StatefulPartitionedCall�$sequential_9/StatefulPartitionedCall�
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_1909425sequential_9_1909427sequential_9_1909429sequential_9_1909431sequential_9_1909433sequential_9_1909435sequential_9_1909437sequential_9_1909439sequential_9_1909441sequential_9_1909443sequential_9_1909445sequential_9_1909447sequential_9_1909449sequential_9_1909451*
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_19089082&
$sequential_9/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall-sequential_9/StatefulPartitionedCall:output:0dense_29_1909466dense_29_1909468*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_19094652"
 dense_29/StatefulPartitionedCall�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1909433*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1909445*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1909449* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentity)dense_29/StatefulPartitionedCall:output:03^conv2d_27/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp!^dense_29/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911069

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
�
�
E__inference_dense_27_layer_call_and_return_conditional_losses_1908846

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�
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
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�
@__inference_CNN_layer_call_and_return_conditional_losses_1910279
input_1H
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_27_conv2d_readvariableop_resource: D
6sequential_9_conv2d_27_biasadd_readvariableop_resource: P
5sequential_9_conv2d_28_conv2d_readvariableop_resource: �E
6sequential_9_conv2d_28_biasadd_readvariableop_resource:	�Q
5sequential_9_conv2d_29_conv2d_readvariableop_resource:��E
6sequential_9_conv2d_29_biasadd_readvariableop_resource:	�I
4sequential_9_dense_27_matmul_readvariableop_resource:���D
5sequential_9_dense_27_biasadd_readvariableop_resource:	�H
4sequential_9_dense_28_matmul_readvariableop_resource:
��D
5sequential_9_dense_28_biasadd_readvariableop_resource:	�:
'dense_29_matmul_readvariableop_resource:	�6
(dense_29_biasadd_readvariableop_resource:
identity��2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�1sequential_9/batch_normalization_9/AssignNewValue�3sequential_9/batch_normalization_9/AssignNewValue_1�Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�1sequential_9/batch_normalization_9/ReadVariableOp�3sequential_9/batch_normalization_9/ReadVariableOp_1�-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�,sequential_9/conv2d_27/Conv2D/ReadVariableOp�-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�,sequential_9/conv2d_28/Conv2D/ReadVariableOp�-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�,sequential_9/conv2d_29/Conv2D/ReadVariableOp�,sequential_9/dense_27/BiasAdd/ReadVariableOp�+sequential_9/dense_27/MatMul/ReadVariableOp�,sequential_9/dense_28/BiasAdd/ReadVariableOp�+sequential_9/dense_28/MatMul/ReadVariableOp�
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
,sequential_9/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_27/Conv2D/ReadVariableOp�
sequential_9/conv2d_27/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
sequential_9/conv2d_27/Conv2D�
-sequential_9/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp�
sequential_9/conv2d_27/BiasAddBiasAdd&sequential_9/conv2d_27/Conv2D:output:05sequential_9/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2 
sequential_9/conv2d_27/BiasAdd�
sequential_9/conv2d_27/ReluRelu'sequential_9/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
sequential_9/conv2d_27/Relu�
%sequential_9/max_pooling2d_27/MaxPoolMaxPool)sequential_9/conv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_27/MaxPool�
,sequential_9/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02.
,sequential_9/conv2d_28/Conv2D/ReadVariableOp�
sequential_9/conv2d_28/Conv2DConv2D.sequential_9/max_pooling2d_27/MaxPool:output:04sequential_9/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
sequential_9/conv2d_28/Conv2D�
-sequential_9/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp�
sequential_9/conv2d_28/BiasAddBiasAdd&sequential_9/conv2d_28/Conv2D:output:05sequential_9/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2 
sequential_9/conv2d_28/BiasAdd�
sequential_9/conv2d_28/ReluRelu'sequential_9/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
sequential_9/conv2d_28/Relu�
%sequential_9/max_pooling2d_28/MaxPoolMaxPool)sequential_9/conv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_28/MaxPool�
,sequential_9/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02.
,sequential_9/conv2d_29/Conv2D/ReadVariableOp�
sequential_9/conv2d_29/Conv2DConv2D.sequential_9/max_pooling2d_28/MaxPool:output:04sequential_9/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
sequential_9/conv2d_29/Conv2D�
-sequential_9/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp�
sequential_9/conv2d_29/BiasAddBiasAdd&sequential_9/conv2d_29/Conv2D:output:05sequential_9/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
sequential_9/conv2d_29/BiasAdd�
sequential_9/conv2d_29/ReluRelu'sequential_9/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_9/conv2d_29/Relu�
%sequential_9/max_pooling2d_29/MaxPoolMaxPool)sequential_9/conv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_29/MaxPool�
%sequential_9/dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2'
%sequential_9/dropout_27/dropout/Const�
#sequential_9/dropout_27/dropout/MulMul.sequential_9/max_pooling2d_29/MaxPool:output:0.sequential_9/dropout_27/dropout/Const:output:0*
T0*0
_output_shapes
:���������		�2%
#sequential_9/dropout_27/dropout/Mul�
%sequential_9/dropout_27/dropout/ShapeShape.sequential_9/max_pooling2d_29/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_27/dropout/Shape�
<sequential_9/dropout_27/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_27/dropout/Shape:output:0*
T0*0
_output_shapes
:���������		�*
dtype02>
<sequential_9/dropout_27/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=20
.sequential_9/dropout_27/dropout/GreaterEqual/y�
,sequential_9/dropout_27/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_27/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_27/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������		�2.
,sequential_9/dropout_27/dropout/GreaterEqual�
$sequential_9/dropout_27/dropout/CastCast0sequential_9/dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:���������		�2&
$sequential_9/dropout_27/dropout/Cast�
%sequential_9/dropout_27/dropout/Mul_1Mul'sequential_9/dropout_27/dropout/Mul:z:0(sequential_9/dropout_27/dropout/Cast:y:0*
T0*0
_output_shapes
:���������		�2'
%sequential_9/dropout_27/dropout/Mul_1�
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
sequential_9/flatten_9/Const�
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/dropout/Mul_1:z:0%sequential_9/flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_9/flatten_9/Reshape�
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp�
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/MatMul�
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp�
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/BiasAdd�
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_27/Relu�
%sequential_9/dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_28/dropout/Const�
#sequential_9/dropout_28/dropout/MulMul(sequential_9/dense_27/Relu:activations:0.sequential_9/dropout_28/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_28/dropout/Mul�
%sequential_9/dropout_28/dropout/ShapeShape(sequential_9/dense_27/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_28/dropout/Shape�
<sequential_9/dropout_28/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_28/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_28/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_28/dropout/GreaterEqual/y�
,sequential_9/dropout_28/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_28/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_28/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_28/dropout/GreaterEqual�
$sequential_9/dropout_28/dropout/CastCast0sequential_9/dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_28/dropout/Cast�
%sequential_9/dropout_28/dropout/Mul_1Mul'sequential_9/dropout_28/dropout/Mul:z:0(sequential_9/dropout_28/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_28/dropout/Mul_1�
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp�
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/dropout/Mul_1:z:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/MatMul�
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp�
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/BiasAdd�
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_9/dense_28/Relu�
%sequential_9/dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_29/dropout/Const�
#sequential_9/dropout_29/dropout/MulMul(sequential_9/dense_28/Relu:activations:0.sequential_9/dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:����������2%
#sequential_9/dropout_29/dropout/Mul�
%sequential_9/dropout_29/dropout/ShapeShape(sequential_9/dense_28/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_29/dropout/Shape�
<sequential_9/dropout_29/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02>
<sequential_9/dropout_29/dropout/random_uniform/RandomUniform�
.sequential_9/dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_29/dropout/GreaterEqual/y�
,sequential_9/dropout_29/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_29/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2.
,sequential_9/dropout_29/dropout/GreaterEqual�
$sequential_9/dropout_29/dropout/CastCast0sequential_9/dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2&
$sequential_9/dropout_29/dropout/Cast�
%sequential_9/dropout_29/dropout/Mul_1Mul'sequential_9/dropout_29/dropout/Mul:z:0(sequential_9/dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2'
%sequential_9/dropout_29/dropout/Mul_1�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMul)sequential_9/dropout_29/dropout/Mul_1:z:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_29/Softmax�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�	
IdentityIdentitydense_29/Softmax:softmax:03^conv2d_27/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_27/BiasAdd/ReadVariableOp-^sequential_9/conv2d_27/Conv2D/ReadVariableOp.^sequential_9/conv2d_28/BiasAdd/ReadVariableOp-^sequential_9/conv2d_28/Conv2D/ReadVariableOp.^sequential_9/conv2d_29/BiasAdd/ReadVariableOp-^sequential_9/conv2d_29/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_12�
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12^
-sequential_9/conv2d_27/BiasAdd/ReadVariableOp-sequential_9/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_27/Conv2D/ReadVariableOp,sequential_9/conv2d_27/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_28/BiasAdd/ReadVariableOp-sequential_9/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_28/Conv2D/ReadVariableOp,sequential_9/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_29/BiasAdd/ReadVariableOp-sequential_9/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_29/Conv2D/ReadVariableOp,sequential_9/conv2d_29/Conv2D/ReadVariableOp2\
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������KK
!
_user_specified_name	input_1
�
�
F__inference_conv2d_28_layer_call_and_return_conditional_losses_1908789

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
%__inference_CNN_layer_call_fn_1910427
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
@__inference_CNN_layer_call_and_return_conditional_losses_19096302
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
�
�
7__inference_batch_normalization_9_layer_call_fn_1911082

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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_19085702
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
*__inference_dense_27_layer_call_fn_1911263

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
E__inference_dense_27_layer_call_and_return_conditional_losses_19088462
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
�
7__inference_batch_normalization_9_layer_call_fn_1911108

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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_19087442
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
�
e
G__inference_dropout_29_layer_call_and_return_conditional_losses_1911327

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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1908614

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
�2
�
@__inference_CNN_layer_call_and_return_conditional_losses_1909630

inputs"
sequential_9_1909577:"
sequential_9_1909579:"
sequential_9_1909581:"
sequential_9_1909583:.
sequential_9_1909585: "
sequential_9_1909587: /
sequential_9_1909589: �#
sequential_9_1909591:	�0
sequential_9_1909593:��#
sequential_9_1909595:	�)
sequential_9_1909597:���#
sequential_9_1909599:	�(
sequential_9_1909601:
��#
sequential_9_1909603:	�#
dense_29_1909606:	�
dense_29_1909608:
identity��2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp� dense_29/StatefulPartitionedCall�$sequential_9/StatefulPartitionedCall�
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_1909577sequential_9_1909579sequential_9_1909581sequential_9_1909583sequential_9_1909585sequential_9_1909587sequential_9_1909589sequential_9_1909591sequential_9_1909593sequential_9_1909595sequential_9_1909597sequential_9_1909599sequential_9_1909601sequential_9_1909603*
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_19092262&
$sequential_9/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall-sequential_9/StatefulPartitionedCall:output:0dense_29_1909606dense_29_1909608*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_19094652"
 dense_29/StatefulPartitionedCall�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1909585*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1909597*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1909601* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentity)dense_29/StatefulPartitionedCall:output:03^conv2d_27/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp!^dense_29/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������KK: : : : : : : : : : : : : : : : 2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�
H
,__inference_dropout_27_layer_call_fn_1911215

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
G__inference_dropout_27_layer_call_and_return_conditional_losses_19088192
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
�
e
G__inference_dropout_29_layer_call_and_return_conditional_losses_1908887

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
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910528

inputs;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_27_conv2d_readvariableop_resource: 7
)conv2d_27_biasadd_readvariableop_resource: C
(conv2d_28_conv2d_readvariableop_resource: �8
)conv2d_28_biasadd_readvariableop_resource:	�D
(conv2d_29_conv2d_readvariableop_resource:��8
)conv2d_29_biasadd_readvariableop_resource:	�<
'dense_27_matmul_readvariableop_resource:���7
(dense_27_biasadd_readvariableop_resource:	�;
'dense_28_matmul_readvariableop_resource:
��7
(dense_28_biasadd_readvariableop_resource:	�
identity��5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp�2conv2d_27/kernel/Regularizer/Square/ReadVariableOp� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�1dense_27/kernel/Regularizer/Square/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�1dense_28/kernel/Regularizer/Square/ReadVariableOp�
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
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK *
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK 2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������KK 2
conv2d_27/Relu�
max_pooling2d_27/MaxPoolMaxPoolconv2d_27/Relu:activations:0*/
_output_shapes
:���������%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2DConv2D!max_pooling2d_27/MaxPool:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�*
paddingSAME*
strides
2
conv2d_28/Conv2D�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������%%�2
conv2d_28/BiasAdd
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:���������%%�2
conv2d_28/Relu�
max_pooling2d_28/MaxPoolMaxPoolconv2d_28/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_28/MaxPool�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_29/Conv2D�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_29/BiasAdd
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_29/Relu�
max_pooling2d_29/MaxPoolMaxPoolconv2d_29/Relu:activations:0*0
_output_shapes
:���������		�*
ksize
*
paddingVALID*
strides
2
max_pooling2d_29/MaxPool�
dropout_27/IdentityIdentity!max_pooling2d_29/MaxPool:output:0*
T0*0
_output_shapes
:���������		�2
dropout_27/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� Q  2
flatten_9/Const�
flatten_9/ReshapeReshapedropout_27/Identity:output:0flatten_9/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_9/Reshape�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_27/Relu�
dropout_28/IdentityIdentitydense_27/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_28/Identity�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_28/MatMul/ReadVariableOp�
dense_28/MatMulMatMuldropout_28/Identity:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_28/MatMul�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_28/BiasAdd/ReadVariableOp�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_28/Relu�
dropout_29/IdentityIdentitydense_28/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_29/Identity�
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp�
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_27/kernel/Regularizer/Square�
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const�
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum�
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2$
"conv2d_27/kernel/Regularizer/mul/x�
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul�
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp�
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:���2$
"dense_27/kernel/Regularizer/Square�
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const�
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/Sum�
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_27/kernel/Regularizer/mul/x�
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul�
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp�
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2$
"dense_28/kernel/Regularizer/Square�
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const�
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/Sum�
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!dense_28/kernel/Regularizer/mul/x�
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul�
IdentityIdentitydropout_29/Identity:output:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������KK: : : : : : : : : : : : : : 2n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������KK
 
_user_specified_nameinputs
�j
�
 __inference__traced_save_1911564
file_prefix.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_m_read_readvariableop6
2savev2_adam_conv2d_27_kernel_m_read_readvariableop4
0savev2_adam_conv2d_27_bias_m_read_readvariableop6
2savev2_adam_conv2d_28_kernel_m_read_readvariableop4
0savev2_adam_conv2d_28_bias_m_read_readvariableop6
2savev2_adam_conv2d_29_kernel_m_read_readvariableop4
0savev2_adam_conv2d_29_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_v_read_readvariableop6
2savev2_adam_conv2d_27_kernel_v_read_readvariableop4
0savev2_adam_conv2d_27_bias_v_read_readvariableop6
2savev2_adam_conv2d_28_kernel_v_read_readvariableop4
0savev2_adam_conv2d_28_bias_v_read_readvariableop6
2savev2_adam_conv2d_29_kernel_v_read_readvariableop4
0savev2_adam_conv2d_29_bias_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop2savev2_adam_conv2d_27_kernel_m_read_readvariableop0savev2_adam_conv2d_27_bias_m_read_readvariableop2savev2_adam_conv2d_28_kernel_m_read_readvariableop0savev2_adam_conv2d_28_bias_m_read_readvariableop2savev2_adam_conv2d_29_kernel_m_read_readvariableop0savev2_adam_conv2d_29_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop2savev2_adam_conv2d_27_kernel_v_read_readvariableop0savev2_adam_conv2d_27_bias_v_read_readvariableop2savev2_adam_conv2d_28_kernel_v_read_readvariableop0savev2_adam_conv2d_28_bias_v_read_readvariableop2savev2_adam_conv2d_29_kernel_v_read_readvariableop0savev2_adam_conv2d_29_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�
N
2__inference_max_pooling2d_27_layer_call_fn_1908686

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
M__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_19086802
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
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1910987

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
	variables
regularization_losses
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
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�d
_tf_keras_sequential�d{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_9_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_27", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_28", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_29", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_9_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_9_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_27", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_28", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_29", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem�m�&m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�v�v�&v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�"
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
 "
trackable_list_wrapper
�
4non_trainable_variables
trainable_variables
5layer_metrics
	variables
6layer_regularization_losses

7layers
8metrics
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
9trainable_variables
:	variables
;regularization_losses
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "lambda_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
�

=axis
	&gamma
'beta
2moving_mean
3moving_variance
>trainable_variables
?	variables
@regularization_losses
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�

(kernel
)bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
�
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_27", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
�


*kernel
+bias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
�
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_28", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
�


,kernel
-bias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
�
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_29", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
�
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
�
^trainable_variables
_	variables
`regularization_losses
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
�	

.kernel
/bias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
�
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
�	

0kernel
1bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
�
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_29", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
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
8
�0
�1
�2"
trackable_list_wrapper
�
rnon_trainable_variables
trainable_variables
slayer_metrics
	variables
tlayer_regularization_losses

ulayers
vmetrics
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_29/kernel
:2dense_29/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables
trainable_variables
xlayer_metrics
	variables
ylayer_regularization_losses

zlayers
{metrics
regularization_losses
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
):'2batch_normalization_9/gamma
(:&2batch_normalization_9/beta
*:( 2conv2d_27/kernel
: 2conv2d_27/bias
+:) �2conv2d_28/kernel
:�2conv2d_28/bias
,:*��2conv2d_29/kernel
:�2conv2d_29/bias
$:"���2dense_27/kernel
:�2dense_27/bias
#:!
��2dense_28/kernel
:�2dense_28/bias
1:/ (2!batch_normalization_9/moving_mean
5:3 (2%batch_normalization_9/moving_variance
.
20
31"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
~non_trainable_variables
9trainable_variables
layer_metrics
:	variables
 �layer_regularization_losses
�layers
�metrics
;regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
�
�non_trainable_variables
>trainable_variables
�layer_metrics
?	variables
 �layer_regularization_losses
�layers
�metrics
@regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
Btrainable_variables
�layer_metrics
C	variables
 �layer_regularization_losses
�layers
�metrics
Dregularization_losses
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
Ftrainable_variables
�layer_metrics
G	variables
 �layer_regularization_losses
�layers
�metrics
Hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Jtrainable_variables
�layer_metrics
K	variables
 �layer_regularization_losses
�layers
�metrics
Lregularization_losses
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
Ntrainable_variables
�layer_metrics
O	variables
 �layer_regularization_losses
�layers
�metrics
Pregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Rtrainable_variables
�layer_metrics
S	variables
 �layer_regularization_losses
�layers
�metrics
Tregularization_losses
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
Vtrainable_variables
�layer_metrics
W	variables
 �layer_regularization_losses
�layers
�metrics
Xregularization_losses
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
Ztrainable_variables
�layer_metrics
[	variables
 �layer_regularization_losses
�layers
�metrics
\regularization_losses
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
^trainable_variables
�layer_metrics
_	variables
 �layer_regularization_losses
�layers
�metrics
`regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
btrainable_variables
�layer_metrics
c	variables
 �layer_regularization_losses
�layers
�metrics
dregularization_losses
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
ftrainable_variables
�layer_metrics
g	variables
 �layer_regularization_losses
�layers
�metrics
hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
jtrainable_variables
�layer_metrics
k	variables
 �layer_regularization_losses
�layers
�metrics
lregularization_losses
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
ntrainable_variables
�layer_metrics
o	variables
 �layer_regularization_losses
�layers
�metrics
pregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
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
':%	�2Adam/dense_29/kernel/m
 :2Adam/dense_29/bias/m
.:,2"Adam/batch_normalization_9/gamma/m
-:+2!Adam/batch_normalization_9/beta/m
/:- 2Adam/conv2d_27/kernel/m
!: 2Adam/conv2d_27/bias/m
0:. �2Adam/conv2d_28/kernel/m
": �2Adam/conv2d_28/bias/m
1:/��2Adam/conv2d_29/kernel/m
": �2Adam/conv2d_29/bias/m
):'���2Adam/dense_27/kernel/m
!:�2Adam/dense_27/bias/m
(:&
��2Adam/dense_28/kernel/m
!:�2Adam/dense_28/bias/m
':%	�2Adam/dense_29/kernel/v
 :2Adam/dense_29/bias/v
.:,2"Adam/batch_normalization_9/gamma/v
-:+2!Adam/batch_normalization_9/beta/v
/:- 2Adam/conv2d_27/kernel/v
!: 2Adam/conv2d_27/bias/v
0:. �2Adam/conv2d_28/kernel/v
": �2Adam/conv2d_28/bias/v
1:/��2Adam/conv2d_29/kernel/v
": �2Adam/conv2d_29/bias/v
):'���2Adam/dense_27/kernel/v
!:�2Adam/dense_27/bias/v
(:&
��2Adam/dense_28/kernel/v
!:�2Adam/dense_28/bias/v
�2�
@__inference_CNN_layer_call_and_return_conditional_losses_1909967
@__inference_CNN_layer_call_and_return_conditional_losses_1910078
@__inference_CNN_layer_call_and_return_conditional_losses_1910168
@__inference_CNN_layer_call_and_return_conditional_losses_1910279�
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
"__inference__wrapped_model_1908548�
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
%__inference_CNN_layer_call_fn_1910316
%__inference_CNN_layer_call_fn_1910353
%__inference_CNN_layer_call_fn_1910390
%__inference_CNN_layer_call_fn_1910427�
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
__inference_call_1790767
__inference_call_1790839
__inference_call_1790911�
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910528
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910632
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910715
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910819�
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
.__inference_sequential_9_layer_call_fn_1910852
.__inference_sequential_9_layer_call_fn_1910885
.__inference_sequential_9_layer_call_fn_1910918
.__inference_sequential_9_layer_call_fn_1910951�
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
E__inference_dense_29_layer_call_and_return_conditional_losses_1910962�
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
*__inference_dense_29_layer_call_fn_1910971�
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
%__inference_signature_wrapper_1909877input_1"�
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
E__inference_lambda_9_layer_call_and_return_conditional_losses_1910979
E__inference_lambda_9_layer_call_and_return_conditional_losses_1910987�
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
*__inference_lambda_9_layer_call_fn_1910992
*__inference_lambda_9_layer_call_fn_1910997�
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911015
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911033
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911051
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911069�
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
7__inference_batch_normalization_9_layer_call_fn_1911082
7__inference_batch_normalization_9_layer_call_fn_1911095
7__inference_batch_normalization_9_layer_call_fn_1911108
7__inference_batch_normalization_9_layer_call_fn_1911121�
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_1911144�
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
+__inference_conv2d_27_layer_call_fn_1911153�
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
M__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_1908680�
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
2__inference_max_pooling2d_27_layer_call_fn_1908686�
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
F__inference_conv2d_28_layer_call_and_return_conditional_losses_1911164�
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
+__inference_conv2d_28_layer_call_fn_1911173�
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
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1908692�
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
2__inference_max_pooling2d_28_layer_call_fn_1908698�
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
F__inference_conv2d_29_layer_call_and_return_conditional_losses_1911184�
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
+__inference_conv2d_29_layer_call_fn_1911193�
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
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1908704�
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
2__inference_max_pooling2d_29_layer_call_fn_1908710�
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
G__inference_dropout_27_layer_call_and_return_conditional_losses_1911198
G__inference_dropout_27_layer_call_and_return_conditional_losses_1911210�
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
,__inference_dropout_27_layer_call_fn_1911215
,__inference_dropout_27_layer_call_fn_1911220�
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
F__inference_flatten_9_layer_call_and_return_conditional_losses_1911226�
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
+__inference_flatten_9_layer_call_fn_1911231�
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
E__inference_dense_27_layer_call_and_return_conditional_losses_1911254�
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
*__inference_dense_27_layer_call_fn_1911263�
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
G__inference_dropout_28_layer_call_and_return_conditional_losses_1911268
G__inference_dropout_28_layer_call_and_return_conditional_losses_1911280�
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
,__inference_dropout_28_layer_call_fn_1911285
,__inference_dropout_28_layer_call_fn_1911290�
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
E__inference_dense_28_layer_call_and_return_conditional_losses_1911313�
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
*__inference_dense_28_layer_call_fn_1911322�
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
G__inference_dropout_29_layer_call_and_return_conditional_losses_1911327
G__inference_dropout_29_layer_call_and_return_conditional_losses_1911339�
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
,__inference_dropout_29_layer_call_fn_1911344
,__inference_dropout_29_layer_call_fn_1911349�
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
__inference_loss_fn_0_1911360�
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
__inference_loss_fn_1_1911371�
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
__inference_loss_fn_2_1911382�
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
@__inference_CNN_layer_call_and_return_conditional_losses_1909967v&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p 
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_1910078v&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_1910168w&'23()*+,-./01<�9
2�/
)�&
input_1���������KK
p 
� "%�"
�
0���������
� �
@__inference_CNN_layer_call_and_return_conditional_losses_1910279w&'23()*+,-./01<�9
2�/
)�&
input_1���������KK
p
� "%�"
�
0���������
� �
%__inference_CNN_layer_call_fn_1910316j&'23()*+,-./01<�9
2�/
)�&
input_1���������KK
p 
� "�����������
%__inference_CNN_layer_call_fn_1910353i&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p 
� "�����������
%__inference_CNN_layer_call_fn_1910390i&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p
� "�����������
%__inference_CNN_layer_call_fn_1910427j&'23()*+,-./01<�9
2�/
)�&
input_1���������KK
p
� "�����������
"__inference__wrapped_model_1908548�&'23()*+,-./018�5
.�+
)�&
input_1���������KK
� "3�0
.
output_1"�
output_1����������
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911015�&'23M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911033�&'23M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911051r&'23;�8
1�.
(�%
inputs���������KK
p 
� "-�*
#� 
0���������KK
� �
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1911069r&'23;�8
1�.
(�%
inputs���������KK
p
� "-�*
#� 
0���������KK
� �
7__inference_batch_normalization_9_layer_call_fn_1911082�&'23M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
7__inference_batch_normalization_9_layer_call_fn_1911095�&'23M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
7__inference_batch_normalization_9_layer_call_fn_1911108e&'23;�8
1�.
(�%
inputs���������KK
p 
� " ����������KK�
7__inference_batch_normalization_9_layer_call_fn_1911121e&'23;�8
1�.
(�%
inputs���������KK
p
� " ����������KKu
__inference_call_1790767Y&'23()*+,-./013�0
)�&
 �
inputs�KK
p
� "�	�u
__inference_call_1790839Y&'23()*+,-./013�0
)�&
 �
inputs�KK
p 
� "�	��
__inference_call_1790911i&'23()*+,-./01;�8
1�.
(�%
inputs���������KK
p 
� "�����������
F__inference_conv2d_27_layer_call_and_return_conditional_losses_1911144l()7�4
-�*
(�%
inputs���������KK
� "-�*
#� 
0���������KK 
� �
+__inference_conv2d_27_layer_call_fn_1911153_()7�4
-�*
(�%
inputs���������KK
� " ����������KK �
F__inference_conv2d_28_layer_call_and_return_conditional_losses_1911164m*+7�4
-�*
(�%
inputs���������%% 
� ".�+
$�!
0���������%%�
� �
+__inference_conv2d_28_layer_call_fn_1911173`*+7�4
-�*
(�%
inputs���������%% 
� "!����������%%��
F__inference_conv2d_29_layer_call_and_return_conditional_losses_1911184n,-8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_conv2d_29_layer_call_fn_1911193a,-8�5
.�+
)�&
inputs����������
� "!������������
E__inference_dense_27_layer_call_and_return_conditional_losses_1911254_./1�.
'�$
"�
inputs�����������
� "&�#
�
0����������
� �
*__inference_dense_27_layer_call_fn_1911263R./1�.
'�$
"�
inputs�����������
� "������������
E__inference_dense_28_layer_call_and_return_conditional_losses_1911313^010�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_28_layer_call_fn_1911322Q010�-
&�#
!�
inputs����������
� "������������
E__inference_dense_29_layer_call_and_return_conditional_losses_1910962]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_29_layer_call_fn_1910971P0�-
&�#
!�
inputs����������
� "�����������
G__inference_dropout_27_layer_call_and_return_conditional_losses_1911198n<�9
2�/
)�&
inputs���������		�
p 
� ".�+
$�!
0���������		�
� �
G__inference_dropout_27_layer_call_and_return_conditional_losses_1911210n<�9
2�/
)�&
inputs���������		�
p
� ".�+
$�!
0���������		�
� �
,__inference_dropout_27_layer_call_fn_1911215a<�9
2�/
)�&
inputs���������		�
p 
� "!����������		��
,__inference_dropout_27_layer_call_fn_1911220a<�9
2�/
)�&
inputs���������		�
p
� "!����������		��
G__inference_dropout_28_layer_call_and_return_conditional_losses_1911268^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_28_layer_call_and_return_conditional_losses_1911280^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_28_layer_call_fn_1911285Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_28_layer_call_fn_1911290Q4�1
*�'
!�
inputs����������
p
� "������������
G__inference_dropout_29_layer_call_and_return_conditional_losses_1911327^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_dropout_29_layer_call_and_return_conditional_losses_1911339^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
,__inference_dropout_29_layer_call_fn_1911344Q4�1
*�'
!�
inputs����������
p 
� "������������
,__inference_dropout_29_layer_call_fn_1911349Q4�1
*�'
!�
inputs����������
p
� "������������
F__inference_flatten_9_layer_call_and_return_conditional_losses_1911226c8�5
.�+
)�&
inputs���������		�
� "'�$
�
0�����������
� �
+__inference_flatten_9_layer_call_fn_1911231V8�5
.�+
)�&
inputs���������		�
� "�������������
E__inference_lambda_9_layer_call_and_return_conditional_losses_1910979p?�<
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
E__inference_lambda_9_layer_call_and_return_conditional_losses_1910987p?�<
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
*__inference_lambda_9_layer_call_fn_1910992c?�<
5�2
(�%
inputs���������KK

 
p 
� " ����������KK�
*__inference_lambda_9_layer_call_fn_1910997c?�<
5�2
(�%
inputs���������KK

 
p
� " ����������KK<
__inference_loss_fn_0_1911360(�

� 
� "� <
__inference_loss_fn_1_1911371.�

� 
� "� <
__inference_loss_fn_2_19113820�

� 
� "� �
M__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_1908680�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_27_layer_call_fn_1908686�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1908692�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_28_layer_call_fn_1908698�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1908704�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_29_layer_call_fn_1908710�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910528y&'23()*+,-./01?�<
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910632y&'23()*+,-./01?�<
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910715�&'23()*+,-./01G�D
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_1910819�&'23()*+,-./01G�D
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
.__inference_sequential_9_layer_call_fn_1910852t&'23()*+,-./01G�D
=�:
0�-
lambda_9_input���������KK
p 

 
� "������������
.__inference_sequential_9_layer_call_fn_1910885l&'23()*+,-./01?�<
5�2
(�%
inputs���������KK
p 

 
� "������������
.__inference_sequential_9_layer_call_fn_1910918l&'23()*+,-./01?�<
5�2
(�%
inputs���������KK
p

 
� "������������
.__inference_sequential_9_layer_call_fn_1910951t&'23()*+,-./01G�D
=�:
0�-
lambda_9_input���������KK
p

 
� "������������
%__inference_signature_wrapper_1909877�&'23()*+,-./01C�@
� 
9�6
4
input_1)�&
input_1���������KK"3�0
.
output_1"�
output_1���������