ѓљ$
С┤
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
Џ
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
Щ
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
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
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
ѓ
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
delete_old_dirsbool(ѕ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718■Э
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	ђ*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
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
ј
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma
Є
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0
ї
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta
Ё
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0
ё
conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
: *
dtype0
t
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
: *
dtype0
Ё
conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*!
shared_nameconv2d_36/kernel
~
$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*'
_output_shapes
: ђ*
dtype0
u
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_36/bias
n
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_37/kernel

$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_38/kernel

$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_38/bias
n
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_39/kernel

$conv2d_39/kernel/Read/ReadVariableOpReadVariableOpconv2d_39/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_39/bias
n
"conv2d_39/bias/Read/ReadVariableOpReadVariableOpconv2d_39/bias*
_output_shapes	
:ђ*
dtype0
|
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђ ђ* 
shared_namedense_21/kernel
u
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel* 
_output_shapes
:
ђ ђ*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:ђ*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
ђђ*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:ђ*
dtype0
џ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean
Њ
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
б
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance
Џ
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
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
Ѕ
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_23/kernel/m
ѓ
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/m
Ћ
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/m
Њ
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_35/kernel/m
І
+Adam/conv2d_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/m*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_35/bias/m
{
)Adam/conv2d_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/m*
_output_shapes
: *
dtype0
Њ
Adam/conv2d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*(
shared_nameAdam/conv2d_36/kernel/m
ї
+Adam/conv2d_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/m*'
_output_shapes
: ђ*
dtype0
Ѓ
Adam/conv2d_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_36/bias/m
|
)Adam/conv2d_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_37/kernel/m
Ї
+Adam/conv2d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_37/bias/m
|
)Adam/conv2d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_38/kernel/m
Ї
+Adam/conv2d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_38/bias/m
|
)Adam/conv2d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_39/kernel/m
Ї
+Adam/conv2d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_39/bias/m
|
)Adam/conv2d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/m*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђ ђ*'
shared_nameAdam/dense_21/kernel/m
Ѓ
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m* 
_output_shapes
:
ђ ђ*
dtype0
Ђ
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_21/bias/m
z
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_22/kernel/m
Ѓ
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_22/bias/m
z
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_23/kernel/v
ѓ
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/v
Ћ
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/v
Њ
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_35/kernel/v
І
+Adam/conv2d_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/v*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_35/bias/v
{
)Adam/conv2d_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/v*
_output_shapes
: *
dtype0
Њ
Adam/conv2d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*(
shared_nameAdam/conv2d_36/kernel/v
ї
+Adam/conv2d_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/v*'
_output_shapes
: ђ*
dtype0
Ѓ
Adam/conv2d_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_36/bias/v
|
)Adam/conv2d_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_37/kernel/v
Ї
+Adam/conv2d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_37/bias/v
|
)Adam/conv2d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_38/kernel/v
Ї
+Adam/conv2d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_38/bias/v
|
)Adam/conv2d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_39/kernel/v
Ї
+Adam/conv2d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_39/bias/v
|
)Adam/conv2d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/v*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђ ђ*'
shared_nameAdam/dense_21/kernel/v
Ѓ
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v* 
_output_shapes
:
ђ ђ*
dtype0
Ђ
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_21/bias/v
z
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_22/kernel/v
Ѓ
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_22/bias/v
z
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes	
:ђ*
dtype0

NoOpNoOp
Лw
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*їw
valueѓwB v BЭv
і

h2ptjl
_output
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
ћ
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
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
е
%iter

&beta_1

'beta_2
	(decay
)learning_ratemщ mЩ*mч+mЧ,m§-m■.m /mђ0mЂ1mѓ2mЃ3mё4mЁ5mє6mЄ7mѕ8mЅ9mіvІ vї*vЇ+vј,vЈ-vљ.vЉ/vњ0vЊ1vћ2vЋ3vќ4vЌ5vў6vЎ7vџ8vЏ9vю
 
є
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
16
 17
ќ
*0
+1
:2
;3
,4
-5
.6
/7
08
19
210
311
412
513
614
715
816
917
18
 19
Г
<metrics
=layer_metrics

>layers
?layer_regularization_losses
regularization_losses
trainable_variables
	variables
@non_trainable_variables
 
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
Ќ
Eaxis
	*gamma
+beta
:moving_mean
;moving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
h

,kernel
-bias
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

.kernel
/bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
R
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
h

0kernel
1bias
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

2kernel
3bias
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

4kernel
5bias
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
R
nregularization_losses
otrainable_variables
p	variables
q	keras_api
R
rregularization_losses
strainable_variables
t	variables
u	keras_api
R
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
h

6kernel
7bias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
T
~regularization_losses
trainable_variables
ђ	variables
Ђ	keras_api
l

8kernel
9bias
ѓregularization_losses
Ѓtrainable_variables
ё	variables
Ё	keras_api
V
єregularization_losses
Єtrainable_variables
ѕ	variables
Ѕ	keras_api
 
v
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
є
*0
+1
:2
;3
,4
-5
.6
/7
08
19
210
311
412
513
614
715
816
917
▓
іmetrics
Іlayer_metrics
їlayers
 Їlayer_regularization_losses
regularization_losses
trainable_variables
	variables
јnon_trainable_variables
NL
VARIABLE_VALUEdense_23/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_23/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
▓
Јmetrics
љlayers
Љlayer_metrics
 њlayer_regularization_losses
!regularization_losses
"trainable_variables
#	variables
Њnon_trainable_variables
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
VARIABLE_VALUEbatch_normalization_7/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_7/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_35/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_35/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_36/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_36/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_37/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_37/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_38/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_38/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_39/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_39/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_21/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_21/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_22/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_22/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_7/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_7/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE

ћ0
Ћ1
 

0
1
 

:0
;1
 
 
 
▓
ќmetrics
Ќlayers
ўlayer_metrics
 Ўlayer_regularization_losses
Aregularization_losses
Btrainable_variables
C	variables
џnon_trainable_variables
 
 

*0
+1

*0
+1
:2
;3
▓
Џmetrics
юlayers
Юlayer_metrics
 ъlayer_regularization_losses
Fregularization_losses
Gtrainable_variables
H	variables
Ъnon_trainable_variables
 

,0
-1

,0
-1
▓
аmetrics
Аlayers
бlayer_metrics
 Бlayer_regularization_losses
Jregularization_losses
Ktrainable_variables
L	variables
цnon_trainable_variables
 
 
 
▓
Цmetrics
дlayers
Дlayer_metrics
 еlayer_regularization_losses
Nregularization_losses
Otrainable_variables
P	variables
Еnon_trainable_variables
 

.0
/1

.0
/1
▓
фmetrics
Фlayers
гlayer_metrics
 Гlayer_regularization_losses
Rregularization_losses
Strainable_variables
T	variables
«non_trainable_variables
 
 
 
▓
»metrics
░layers
▒layer_metrics
 ▓layer_regularization_losses
Vregularization_losses
Wtrainable_variables
X	variables
│non_trainable_variables
 

00
11

00
11
▓
┤metrics
хlayers
Хlayer_metrics
 иlayer_regularization_losses
Zregularization_losses
[trainable_variables
\	variables
Иnon_trainable_variables
 
 
 
▓
╣metrics
║layers
╗layer_metrics
 ╝layer_regularization_losses
^regularization_losses
_trainable_variables
`	variables
йnon_trainable_variables
 

20
31

20
31
▓
Йmetrics
┐layers
└layer_metrics
 ┴layer_regularization_losses
bregularization_losses
ctrainable_variables
d	variables
┬non_trainable_variables
 
 
 
▓
├metrics
─layers
┼layer_metrics
 кlayer_regularization_losses
fregularization_losses
gtrainable_variables
h	variables
Кnon_trainable_variables
 

40
51

40
51
▓
╚metrics
╔layers
╩layer_metrics
 ╦layer_regularization_losses
jregularization_losses
ktrainable_variables
l	variables
╠non_trainable_variables
 
 
 
▓
═metrics
╬layers
¤layer_metrics
 лlayer_regularization_losses
nregularization_losses
otrainable_variables
p	variables
Лnon_trainable_variables
 
 
 
▓
мmetrics
Мlayers
нlayer_metrics
 Нlayer_regularization_losses
rregularization_losses
strainable_variables
t	variables
оnon_trainable_variables
 
 
 
▓
Оmetrics
пlayers
┘layer_metrics
 ┌layer_regularization_losses
vregularization_losses
wtrainable_variables
x	variables
█non_trainable_variables
 

60
71

60
71
▓
▄metrics
Пlayers
яlayer_metrics
 ▀layer_regularization_losses
zregularization_losses
{trainable_variables
|	variables
Яnon_trainable_variables
 
 
 
│
рmetrics
Рlayers
сlayer_metrics
 Сlayer_regularization_losses
~regularization_losses
trainable_variables
ђ	variables
тnon_trainable_variables
 

80
91

80
91
х
Тmetrics
уlayers
Уlayer_metrics
 жlayer_regularization_losses
ѓregularization_losses
Ѓtrainable_variables
ё	variables
Жnon_trainable_variables
 
 
 
х
вmetrics
Вlayers
ьlayer_metrics
 Ьlayer_regularization_losses
єregularization_losses
Єtrainable_variables
ѕ	variables
№non_trainable_variables
 
 
є
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
:0
;1
 
 
 
 
 
8

­total

ыcount
Ы	variables
з	keras_api
I

Зtotal

шcount
Ш
_fn_kwargs
э	variables
Э	keras_api
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
:0
;1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
­0
ы1

Ы	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

З0
ш1

э	variables
qo
VARIABLE_VALUEAdam/dense_23/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_23/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_35/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_35/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_36/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_36/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_37/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_37/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_38/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_38/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_39/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_39/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_21/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_21/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_22/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_22/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_23/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_23/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_35/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_35/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_36/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_36/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_37/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_37/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_38/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_38/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_39/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_39/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_21/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_21/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_22/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_22/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
і
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
В
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *-
f(R&
$__inference_signature_wrapper_936183
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp+Adam/conv2d_35/kernel/m/Read/ReadVariableOp)Adam/conv2d_35/bias/m/Read/ReadVariableOp+Adam/conv2d_36/kernel/m/Read/ReadVariableOp)Adam/conv2d_36/bias/m/Read/ReadVariableOp+Adam/conv2d_37/kernel/m/Read/ReadVariableOp)Adam/conv2d_37/bias/m/Read/ReadVariableOp+Adam/conv2d_38/kernel/m/Read/ReadVariableOp)Adam/conv2d_38/bias/m/Read/ReadVariableOp+Adam/conv2d_39/kernel/m/Read/ReadVariableOp)Adam/conv2d_39/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp+Adam/conv2d_35/kernel/v/Read/ReadVariableOp)Adam/conv2d_35/bias/v/Read/ReadVariableOp+Adam/conv2d_36/kernel/v/Read/ReadVariableOp)Adam/conv2d_36/bias/v/Read/ReadVariableOp+Adam/conv2d_37/kernel/v/Read/ReadVariableOp)Adam/conv2d_37/bias/v/Read/ReadVariableOp+Adam/conv2d_38/kernel/v/Read/ReadVariableOp)Adam/conv2d_38/bias/v/Read/ReadVariableOp+Adam/conv2d_39/kernel/v/Read/ReadVariableOp)Adam/conv2d_39/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOpConst*N
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
GPU2 *0J 8ѓ *(
f#R!
__inference__traced_save_938138
л
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_7/gammabatch_normalization_7/betaconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/bias!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancetotalcounttotal_1count_1Adam/dense_23/kernel/mAdam/dense_23/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv2d_35/kernel/mAdam/conv2d_35/bias/mAdam/conv2d_36/kernel/mAdam/conv2d_36/bias/mAdam/conv2d_37/kernel/mAdam/conv2d_37/bias/mAdam/conv2d_38/kernel/mAdam/conv2d_38/bias/mAdam/conv2d_39/kernel/mAdam/conv2d_39/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/vAdam/dense_23/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/conv2d_35/kernel/vAdam/conv2d_35/bias/vAdam/conv2d_36/kernel/vAdam/conv2d_36/bias/vAdam/conv2d_37/kernel/vAdam/conv2d_37/bias/vAdam/conv2d_38/kernel/vAdam/conv2d_38/bias/vAdam/conv2d_39/kernel/vAdam/conv2d_39/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/v*M
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
GPU2 *0J 8ѓ *+
f&R$
"__inference__traced_restore_938343ил
Б
Ќ
)__inference_dense_23_layer_call_fn_937458

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_9357072
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ј▓
 
H__inference_sequential_7_layer_call_and_return_conditional_losses_937449
lambda_7_input;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_35_conv2d_readvariableop_resource: 7
)conv2d_35_biasadd_readvariableop_resource: C
(conv2d_36_conv2d_readvariableop_resource: ђ8
)conv2d_36_biasadd_readvariableop_resource:	ђD
(conv2d_37_conv2d_readvariableop_resource:ђђ8
)conv2d_37_biasadd_readvariableop_resource:	ђD
(conv2d_38_conv2d_readvariableop_resource:ђђ8
)conv2d_38_biasadd_readvariableop_resource:	ђD
(conv2d_39_conv2d_readvariableop_resource:ђђ8
)conv2d_39_biasadd_readvariableop_resource:	ђ;
'dense_21_matmul_readvariableop_resource:
ђ ђ7
(dense_21_biasadd_readvariableop_resource:	ђ;
'dense_22_matmul_readvariableop_resource:
ђђ7
(dense_22_biasadd_readvariableop_resource:	ђ
identityѕб$batch_normalization_7/AssignNewValueб&batch_normalization_7/AssignNewValue_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б conv2d_35/BiasAdd/ReadVariableOpбconv2d_35/Conv2D/ReadVariableOpб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб conv2d_36/BiasAdd/ReadVariableOpбconv2d_36/Conv2D/ReadVariableOpб conv2d_37/BiasAdd/ReadVariableOpбconv2d_37/Conv2D/ReadVariableOpб conv2d_38/BiasAdd/ReadVariableOpбconv2d_38/Conv2D/ReadVariableOpб conv2d_39/BiasAdd/ReadVariableOpбconv2d_39/Conv2D/ReadVariableOpбdense_21/BiasAdd/ReadVariableOpбdense_21/MatMul/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЎ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Ў
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2▓
lambda_7/strided_sliceStridedSlicelambda_7_input%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_7/strided_sliceХ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1ж
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_7/FusedBatchNormV3░
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue╝
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1│
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_35/Conv2D/ReadVariableOpт
conv2d_35/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_35/Conv2Dф
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp░
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_35/BiasAdd~
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_35/Relu╩
max_pooling2d_35/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_35/MaxPool┤
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_36/Conv2D/ReadVariableOpП
conv2d_36/Conv2DConv2D!max_pooling2d_35/MaxPool:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_36/Conv2DФ
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp▒
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_36/BiasAdd
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_36/Relu╦
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPoolх
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_37/Conv2D/ReadVariableOpП
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_37/Conv2DФ
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp▒
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_37/Relu╦
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPoolх
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_38/Conv2D/ReadVariableOpП
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
conv2d_38/Conv2DФ
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
conv2d_38/Relu╦
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolх
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_39/Conv2D/ReadVariableOpП
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_39/Conv2DФ
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▒
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_39/Relu╦
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPooly
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_21/dropout/ConstИ
dropout_21/dropout/MulMul!max_pooling2d_39/MaxPool:output:0!dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout_21/dropout/MulЁ
dropout_21/dropout/ShapeShape!max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shapeя
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformІ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_21/dropout/GreaterEqual/yз
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2!
dropout_21/dropout/GreaterEqualЕ
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout_21/dropout/Cast»
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout_21/dropout/Mul_1s
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_7/Constю
flatten_7/ReshapeReshapedropout_21/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2
flatten_7/Reshapeф
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02 
dense_21/MatMul/ReadVariableOpБ
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_21/MatMulе
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_21/BiasAdd/ReadVariableOpд
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_21/BiasAddt
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_21/Reluy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_22/dropout/Constф
dropout_22/dropout/MulMuldense_21/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_22/dropout/Mul
dropout_22/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shapeо
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_22/dropout/random_uniform/RandomUniformІ
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_22/dropout/GreaterEqual/yв
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_22/dropout/GreaterEqualА
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_22/dropout/CastД
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_22/dropout/Mul_1ф
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_22/MatMul/ReadVariableOpЦ
dense_22/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_22/MatMulе
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_22/BiasAdd/ReadVariableOpд
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_22/BiasAddt
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_22/Reluy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_23/dropout/Constф
dropout_23/dropout/MulMuldense_22/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_23/dropout/Mul
dropout_23/dropout/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shapeо
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_23/dropout/random_uniform/RandomUniformІ
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_23/dropout/GreaterEqual/yв
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_23/dropout/GreaterEqualА
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_23/dropout/CastД
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_23/dropout/Mul_1┘
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulл
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulл
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul 
IdentityIdentitydropout_23/dropout/Mul_1:z:0%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_7_input
┐
│
E__inference_conv2d_35_layer_call_and_return_conditional_losses_937651

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu¤
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ю
ђ
E__inference_conv2d_36_layer_call_and_return_conditional_losses_937671

inputs9
conv2d_readvariableop_resource: ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         %% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         %% 
 
_user_specified_nameinputs
╣
г
D__inference_dense_22_layer_call_and_return_conditional_losses_935022

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
ReluК
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Бm
№	
H__inference_sequential_7_layer_call_and_return_conditional_losses_935054

inputs*
batch_normalization_7_934855:*
batch_normalization_7_934857:*
batch_normalization_7_934859:*
batch_normalization_7_934861:*
conv2d_35_934882: 
conv2d_35_934884: +
conv2d_36_934900: ђ
conv2d_36_934902:	ђ,
conv2d_37_934918:ђђ
conv2d_37_934920:	ђ,
conv2d_38_934936:ђђ
conv2d_38_934938:	ђ,
conv2d_39_934954:ђђ
conv2d_39_934956:	ђ#
dense_21_934993:
ђ ђ
dense_21_934995:	ђ#
dense_22_935023:
ђђ
dense_22_935025:	ђ
identityѕб-batch_normalization_7/StatefulPartitionedCallб!conv2d_35/StatefulPartitionedCallб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб!conv2d_36/StatefulPartitionedCallб!conv2d_37/StatefulPartitionedCallб!conv2d_38/StatefulPartitionedCallб!conv2d_39/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб1dense_21/kernel/Regularizer/Square/ReadVariableOpб dense_22/StatefulPartitionedCallб1dense_22/kernel/Regularizer/Square/ReadVariableOpр
lambda_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_lambda_7_layer_call_and_return_conditional_losses_9348352
lambda_7/PartitionedCallй
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0batch_normalization_7_934855batch_normalization_7_934857batch_normalization_7_934859batch_normalization_7_934861*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9348542/
-batch_normalization_7/StatefulPartitionedCallо
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_35_934882conv2d_35_934884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_9348812#
!conv2d_35/StatefulPartitionedCallЮ
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         %% * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_9347662"
 max_pooling2d_35/PartitionedCall╩
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0conv2d_36_934900conv2d_36_934902*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_36_layer_call_and_return_conditional_losses_9348992#
!conv2d_36/StatefulPartitionedCallъ
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_9347782"
 max_pooling2d_36/PartitionedCall╩
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_37_934918conv2d_37_934920*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_37_layer_call_and_return_conditional_losses_9349172#
!conv2d_37/StatefulPartitionedCallъ
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_9347902"
 max_pooling2d_37/PartitionedCall╩
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_38_934936conv2d_38_934938*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_38_layer_call_and_return_conditional_losses_9349352#
!conv2d_38/StatefulPartitionedCallъ
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_9348022"
 max_pooling2d_38/PartitionedCall╩
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_934954conv2d_39_934956*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_39_layer_call_and_return_conditional_losses_9349532#
!conv2d_39/StatefulPartitionedCallъ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_9348142"
 max_pooling2d_39/PartitionedCallІ
dropout_21/PartitionedCallPartitionedCall)max_pooling2d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_9349652
dropout_21/PartitionedCallЩ
flatten_7/PartitionedCallPartitionedCall#dropout_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_9349732
flatten_7/PartitionedCallХ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_21_934993dense_21_934995*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_9349922"
 dense_21/StatefulPartitionedCallЃ
dropout_22/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_9350032
dropout_22/PartitionedCallи
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_22_935023dense_22_935025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_9350222"
 dense_22/StatefulPartitionedCallЃ
dropout_23/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_9350332
dropout_23/PartitionedCall┴
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_934882*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulИ
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21_934993* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulИ
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22_935023* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul┐
IdentityIdentity#dropout_23/PartitionedCall:output:0.^batch_normalization_7/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/Square/ReadVariableOp!^dense_22/StatefulPartitionedCall2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ж
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_934973

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
х
e
F__inference_dropout_23_layer_call_and_return_conditional_losses_935113

inputs
identityѕc
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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
И

Ш
D__inference_dense_23_layer_call_and_return_conditional_losses_937469

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ќ
ѓ
-__inference_sequential_7_layer_call_fn_936970

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ
identityѕбStatefulPartitionedCallМ
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
:         ђ*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_9354202
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_39_layer_call_fn_934820

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_9348142
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
х
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_937828

inputs
identityѕc
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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ЂЉ
С
__inference_call_889683

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_35_conv2d_readvariableop_resource: D
6sequential_7_conv2d_35_biasadd_readvariableop_resource: P
5sequential_7_conv2d_36_conv2d_readvariableop_resource: ђE
6sequential_7_conv2d_36_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_37_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_37_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_38_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_38_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_39_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_39_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_21_matmul_readvariableop_resource:
ђ ђD
5sequential_7_dense_21_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_22_matmul_readvariableop_resource:
ђђD
5sequential_7_dense_22_biasadd_readvariableop_resource:	ђ:
'dense_23_matmul_readvariableop_resource:	ђ6
(dense_23_biasadd_readvariableop_resource:
identityѕбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpбBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_7/batch_normalization_7/ReadVariableOpб3sequential_7/batch_normalization_7/ReadVariableOp_1б-sequential_7/conv2d_35/BiasAdd/ReadVariableOpб,sequential_7/conv2d_35/Conv2D/ReadVariableOpб-sequential_7/conv2d_36/BiasAdd/ReadVariableOpб,sequential_7/conv2d_36/Conv2D/ReadVariableOpб-sequential_7/conv2d_37/BiasAdd/ReadVariableOpб,sequential_7/conv2d_37/Conv2D/ReadVariableOpб-sequential_7/conv2d_38/BiasAdd/ReadVariableOpб,sequential_7/conv2d_38/Conv2D/ReadVariableOpб-sequential_7/conv2d_39/BiasAdd/ReadVariableOpб,sequential_7/conv2d_39/Conv2D/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOp»
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2с
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ђKK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_sliceП
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpс
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1љ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpќ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1║
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:ђKK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_35/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_35/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK *
paddingSAME*
strides
2
sequential_7/conv2d_35/Conv2DЛ
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp▄
sequential_7/conv2d_35/BiasAddBiasAdd&sequential_7/conv2d_35/Conv2D:output:05sequential_7/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK 2 
sequential_7/conv2d_35/BiasAddЮ
sequential_7/conv2d_35/ReluRelu'sequential_7/conv2d_35/BiasAdd:output:0*
T0*'
_output_shapes
:ђKK 2
sequential_7/conv2d_35/Reluж
%sequential_7/max_pooling2d_35/MaxPoolMaxPool)sequential_7/conv2d_35/Relu:activations:0*'
_output_shapes
:ђ%% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_35/MaxPool█
,sequential_7/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_7/conv2d_36/Conv2D/ReadVariableOpЅ
sequential_7/conv2d_36/Conv2DConv2D.sequential_7/max_pooling2d_35/MaxPool:output:04sequential_7/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ*
paddingSAME*
strides
2
sequential_7/conv2d_36/Conv2Dм
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpП
sequential_7/conv2d_36/BiasAddBiasAdd&sequential_7/conv2d_36/Conv2D:output:05sequential_7/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ2 
sequential_7/conv2d_36/BiasAddъ
sequential_7/conv2d_36/ReluRelu'sequential_7/conv2d_36/BiasAdd:output:0*
T0*(
_output_shapes
:ђ%%ђ2
sequential_7/conv2d_36/ReluЖ
%sequential_7/max_pooling2d_36/MaxPoolMaxPool)sequential_7/conv2d_36/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_36/MaxPool▄
,sequential_7/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_37/Conv2D/ReadVariableOpЅ
sequential_7/conv2d_37/Conv2DConv2D.sequential_7/max_pooling2d_36/MaxPool:output:04sequential_7/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_7/conv2d_37/Conv2Dм
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpП
sequential_7/conv2d_37/BiasAddBiasAdd&sequential_7/conv2d_37/Conv2D:output:05sequential_7/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_7/conv2d_37/BiasAddъ
sequential_7/conv2d_37/ReluRelu'sequential_7/conv2d_37/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_7/conv2d_37/ReluЖ
%sequential_7/max_pooling2d_37/MaxPoolMaxPool)sequential_7/conv2d_37/Relu:activations:0*(
_output_shapes
:ђ		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_37/MaxPool▄
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpЅ
sequential_7/conv2d_38/Conv2DConv2D.sequential_7/max_pooling2d_37/MaxPool:output:04sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ		ђ*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2Dм
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpП
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ		ђ2 
sequential_7/conv2d_38/BiasAddъ
sequential_7/conv2d_38/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*(
_output_shapes
:ђ		ђ2
sequential_7/conv2d_38/ReluЖ
%sequential_7/max_pooling2d_38/MaxPoolMaxPool)sequential_7/conv2d_38/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPool▄
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpЅ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2Dм
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpП
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_7/conv2d_39/BiasAddъ
sequential_7/conv2d_39/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_7/conv2d_39/ReluЖ
%sequential_7/max_pooling2d_39/MaxPoolMaxPool)sequential_7/conv2d_39/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPool│
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_39/MaxPool:output:0*
T0*(
_output_shapes
:ђђ2"
 sequential_7/dropout_21/IdentityЇ
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╚
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0* 
_output_shapes
:
ђђ 2 
sequential_7/flatten_7/ReshapeЛ
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp¤
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_21/MatMul¤
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOpм
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_21/BiasAddЊ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_21/ReluЦ
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_7/dropout_22/IdentityЛ
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOpЛ
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_22/MatMul¤
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOpм
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_22/BiasAddЊ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_22/ReluЦ
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_7/dropout_23/IdentityЕ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_23/MatMul/ReadVariableOpЕ
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЮ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_23/BiasAddt
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*
_output_shapes
:	ђ2
dense_23/Softmax┤
IdentityIdentitydense_23/Softmax:softmax:0 ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_35/BiasAdd/ReadVariableOp-^sequential_7/conv2d_35/Conv2D/ReadVariableOp.^sequential_7/conv2d_36/BiasAdd/ReadVariableOp-^sequential_7/conv2d_36/Conv2D/ReadVariableOp.^sequential_7/conv2d_37/BiasAdd/ReadVariableOp-^sequential_7/conv2d_37/Conv2D/ReadVariableOp.^sequential_7/conv2d_38/BiasAdd/ReadVariableOp-^sequential_7/conv2d_38/Conv2D/ReadVariableOp.^sequential_7/conv2d_39/BiasAdd/ReadVariableOp-^sequential_7/conv2d_39/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*
_output_shapes
:	ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ђKK: : : : : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2ѕ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp-sequential_7/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_35/Conv2D/ReadVariableOp,sequential_7/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_36/BiasAdd/ReadVariableOp-sequential_7/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_36/Conv2D/ReadVariableOp,sequential_7/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_37/BiasAdd/ReadVariableOp-sequential_7/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_37/Conv2D/ReadVariableOp,sequential_7/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_38/BiasAdd/ReadVariableOp-sequential_7/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_38/Conv2D/ReadVariableOp,sequential_7/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_39/BiasAdd/ReadVariableOp-sequential_7/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_39/Conv2D/ReadVariableOp,sequential_7/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:O K
'
_output_shapes
:ђKK
 
_user_specified_nameinputs
э
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_937875

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Џ
о
!__inference__wrapped_model_934634
input_1
cnn3_934592:
cnn3_934594:
cnn3_934596:
cnn3_934598:%
cnn3_934600: 
cnn3_934602: &
cnn3_934604: ђ
cnn3_934606:	ђ'
cnn3_934608:ђђ
cnn3_934610:	ђ'
cnn3_934612:ђђ
cnn3_934614:	ђ'
cnn3_934616:ђђ
cnn3_934618:	ђ
cnn3_934620:
ђ ђ
cnn3_934622:	ђ
cnn3_934624:
ђђ
cnn3_934626:	ђ
cnn3_934628:	ђ
cnn3_934630:
identityѕбCNN3/StatefulPartitionedCallв
CNN3/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn3_934592cnn3_934594cnn3_934596cnn3_934598cnn3_934600cnn3_934602cnn3_934604cnn3_934606cnn3_934608cnn3_934610cnn3_934612cnn3_934614cnn3_934616cnn3_934618cnn3_934620cnn3_934622cnn3_934624cnn3_934626cnn3_934628cnn3_934630* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ * 
fR
__inference_call_8872692
CNN3/StatefulPartitionedCallў
IdentityIdentity%CNN3/StatefulPartitionedCall:output:0^CNN3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2<
CNN3/StatefulPartitionedCallCNN3/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
ЌЋ
К)
"__inference__traced_restore_938343
file_prefix3
 assignvariableop_dense_23_kernel:	ђ.
 assignvariableop_1_dense_23_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_7_gamma:;
-assignvariableop_8_batch_normalization_7_beta:=
#assignvariableop_9_conv2d_35_kernel: 0
"assignvariableop_10_conv2d_35_bias: ?
$assignvariableop_11_conv2d_36_kernel: ђ1
"assignvariableop_12_conv2d_36_bias:	ђ@
$assignvariableop_13_conv2d_37_kernel:ђђ1
"assignvariableop_14_conv2d_37_bias:	ђ@
$assignvariableop_15_conv2d_38_kernel:ђђ1
"assignvariableop_16_conv2d_38_bias:	ђ@
$assignvariableop_17_conv2d_39_kernel:ђђ1
"assignvariableop_18_conv2d_39_bias:	ђ7
#assignvariableop_19_dense_21_kernel:
ђ ђ0
!assignvariableop_20_dense_21_bias:	ђ7
#assignvariableop_21_dense_22_kernel:
ђђ0
!assignvariableop_22_dense_22_bias:	ђC
5assignvariableop_23_batch_normalization_7_moving_mean:G
9assignvariableop_24_batch_normalization_7_moving_variance:#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: =
*assignvariableop_29_adam_dense_23_kernel_m:	ђ6
(assignvariableop_30_adam_dense_23_bias_m:D
6assignvariableop_31_adam_batch_normalization_7_gamma_m:C
5assignvariableop_32_adam_batch_normalization_7_beta_m:E
+assignvariableop_33_adam_conv2d_35_kernel_m: 7
)assignvariableop_34_adam_conv2d_35_bias_m: F
+assignvariableop_35_adam_conv2d_36_kernel_m: ђ8
)assignvariableop_36_adam_conv2d_36_bias_m:	ђG
+assignvariableop_37_adam_conv2d_37_kernel_m:ђђ8
)assignvariableop_38_adam_conv2d_37_bias_m:	ђG
+assignvariableop_39_adam_conv2d_38_kernel_m:ђђ8
)assignvariableop_40_adam_conv2d_38_bias_m:	ђG
+assignvariableop_41_adam_conv2d_39_kernel_m:ђђ8
)assignvariableop_42_adam_conv2d_39_bias_m:	ђ>
*assignvariableop_43_adam_dense_21_kernel_m:
ђ ђ7
(assignvariableop_44_adam_dense_21_bias_m:	ђ>
*assignvariableop_45_adam_dense_22_kernel_m:
ђђ7
(assignvariableop_46_adam_dense_22_bias_m:	ђ=
*assignvariableop_47_adam_dense_23_kernel_v:	ђ6
(assignvariableop_48_adam_dense_23_bias_v:D
6assignvariableop_49_adam_batch_normalization_7_gamma_v:C
5assignvariableop_50_adam_batch_normalization_7_beta_v:E
+assignvariableop_51_adam_conv2d_35_kernel_v: 7
)assignvariableop_52_adam_conv2d_35_bias_v: F
+assignvariableop_53_adam_conv2d_36_kernel_v: ђ8
)assignvariableop_54_adam_conv2d_36_bias_v:	ђG
+assignvariableop_55_adam_conv2d_37_kernel_v:ђђ8
)assignvariableop_56_adam_conv2d_37_bias_v:	ђG
+assignvariableop_57_adam_conv2d_38_kernel_v:ђђ8
)assignvariableop_58_adam_conv2d_38_bias_v:	ђG
+assignvariableop_59_adam_conv2d_39_kernel_v:ђђ8
)assignvariableop_60_adam_conv2d_39_bias_v:	ђ>
*assignvariableop_61_adam_dense_21_kernel_v:
ђ ђ7
(assignvariableop_62_adam_dense_21_bias_v:	ђ>
*assignvariableop_63_adam_dense_22_kernel_v:
ђђ7
(assignvariableop_64_adam_dense_22_bias_v:	ђ
identity_66ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9њ"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*ъ!
valueћ!BЉ!BB)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЋ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*Ў
valueЈBїBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЭ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ъ
_output_shapesІ
ѕ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*P
dtypesF
D2B	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЪ
AssignVariableOpAssignVariableOp assignvariableop_dense_23_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_23_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2А
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Б
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Б
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5б
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ф
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7│
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_7_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▓
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_7_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9е
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_35_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ф
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_35_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_36_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ф
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_36_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_37_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ф
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_37_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15г
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_38_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ф
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_38_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17г
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_39_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ф
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_39_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ф
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_21_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Е
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_21_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ф
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_22_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Е
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_22_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23й
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_7_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24┴
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_7_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25А
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26А
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Б
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Б
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▓
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_23_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30░
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_23_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Й
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_7_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32й
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_batch_normalization_7_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_35_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_35_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35│
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_36_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▒
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_36_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37│
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_37_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_37_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39│
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_38_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40▒
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_38_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41│
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_39_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42▒
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_39_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▓
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_21_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44░
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_21_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45▓
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_22_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46░
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_22_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47▓
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_23_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48░
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_23_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Й
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_7_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50й
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_7_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51│
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_35_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52▒
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_35_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53│
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_36_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54▒
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_36_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55│
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_37_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56▒
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_37_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57│
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_38_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58▒
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_38_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59│
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_39_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60▒
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_39_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61▓
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_21_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62░
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_21_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63▓
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_22_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64░
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_22_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_649
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЗ
Identity_65Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_65у
Identity_66IdentityIdentity_65:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_66"#
identity_66Identity_66:output:0*Ў
_input_shapesЄ
ё: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
г
h
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_934790

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
У
│
__inference_loss_fn_1_937909N
:dense_21_kernel_regularizer_square_readvariableop_resource:
ђ ђ
identityѕб1dense_21/kernel/Regularizer/Square/ReadVariableOpс
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_21_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulџ
IdentityIdentity#dense_21/kernel/Regularizer/mul:z:02^dense_21/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp
├
ю
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_934854

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
»
і
-__inference_sequential_7_layer_call_fn_937011
lambda_7_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCalllambda_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђ*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_9354202
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_7_input
я
│
%__inference_CNN3_layer_call_fn_936318

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:
identityѕбStatefulPartitionedCallТ
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
:         *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_CNN3_layer_call_and_return_conditional_losses_9358962
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
А
Ђ
E__inference_conv2d_37_layer_call_and_return_conditional_losses_937691

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ќ
d
F__inference_dropout_21_layer_call_and_return_conditional_losses_937746

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_934766

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▒
і
-__inference_sequential_7_layer_call_fn_936888
lambda_7_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ
identityѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCalllambda_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_9350542
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_7_input
Ђ4
н
@__inference_CNN3_layer_call_and_return_conditional_losses_935732

inputs!
sequential_7_935659:!
sequential_7_935661:!
sequential_7_935663:!
sequential_7_935665:-
sequential_7_935667: !
sequential_7_935669: .
sequential_7_935671: ђ"
sequential_7_935673:	ђ/
sequential_7_935675:ђђ"
sequential_7_935677:	ђ/
sequential_7_935679:ђђ"
sequential_7_935681:	ђ/
sequential_7_935683:ђђ"
sequential_7_935685:	ђ'
sequential_7_935687:
ђ ђ"
sequential_7_935689:	ђ'
sequential_7_935691:
ђђ"
sequential_7_935693:	ђ"
dense_23_935708:	ђ
dense_23_935710:
identityѕб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpб dense_23/StatefulPartitionedCallб$sequential_7/StatefulPartitionedCallъ
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7_935659sequential_7_935661sequential_7_935663sequential_7_935665sequential_7_935667sequential_7_935669sequential_7_935671sequential_7_935673sequential_7_935675sequential_7_935677sequential_7_935679sequential_7_935681sequential_7_935683sequential_7_935685sequential_7_935687sequential_7_935689sequential_7_935691sequential_7_935693*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_9350542&
$sequential_7/StatefulPartitionedCall└
 dense_23/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0dense_23_935708dense_23_935710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_9357072"
 dense_23/StatefulPartitionedCall─
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_935667*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mul╝
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_935687* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╝
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_935691* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulС
IdentityIdentity)dense_23/StatefulPartitionedCall:output:03^conv2d_35/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp!^dense_23/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_934802

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
І
ю
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_934656

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_934778

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┬
`
D__inference_lambda_7_layer_call_and_return_conditional_losses_937495

inputs
identityЃ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЄ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1Є
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2§
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
д
Л
6__inference_batch_normalization_7_layer_call_fn_937534

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9348542
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ш
e
F__inference_dropout_21_layer_call_and_return_conditional_losses_935185

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yК
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
╔
G
+__inference_dropout_22_layer_call_fn_937806

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_9350032
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┬
`
D__inference_lambda_7_layer_call_and_return_conditional_losses_934835

inputs
identityЃ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЄ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1Є
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2§
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
А
Ђ
E__inference_conv2d_38_layer_call_and_return_conditional_losses_937711

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
Ш▒
э
H__inference_sequential_7_layer_call_and_return_conditional_losses_937230

inputs;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_35_conv2d_readvariableop_resource: 7
)conv2d_35_biasadd_readvariableop_resource: C
(conv2d_36_conv2d_readvariableop_resource: ђ8
)conv2d_36_biasadd_readvariableop_resource:	ђD
(conv2d_37_conv2d_readvariableop_resource:ђђ8
)conv2d_37_biasadd_readvariableop_resource:	ђD
(conv2d_38_conv2d_readvariableop_resource:ђђ8
)conv2d_38_biasadd_readvariableop_resource:	ђD
(conv2d_39_conv2d_readvariableop_resource:ђђ8
)conv2d_39_biasadd_readvariableop_resource:	ђ;
'dense_21_matmul_readvariableop_resource:
ђ ђ7
(dense_21_biasadd_readvariableop_resource:	ђ;
'dense_22_matmul_readvariableop_resource:
ђђ7
(dense_22_biasadd_readvariableop_resource:	ђ
identityѕб$batch_normalization_7/AssignNewValueб&batch_normalization_7/AssignNewValue_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б conv2d_35/BiasAdd/ReadVariableOpбconv2d_35/Conv2D/ReadVariableOpб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб conv2d_36/BiasAdd/ReadVariableOpбconv2d_36/Conv2D/ReadVariableOpб conv2d_37/BiasAdd/ReadVariableOpбconv2d_37/Conv2D/ReadVariableOpб conv2d_38/BiasAdd/ReadVariableOpбconv2d_38/Conv2D/ReadVariableOpб conv2d_39/BiasAdd/ReadVariableOpбconv2d_39/Conv2D/ReadVariableOpбdense_21/BiasAdd/ReadVariableOpбdense_21/MatMul/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЎ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Ў
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2ф
lambda_7/strided_sliceStridedSliceinputs%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_7/strided_sliceХ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1ж
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_7/FusedBatchNormV3░
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue╝
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1│
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_35/Conv2D/ReadVariableOpт
conv2d_35/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_35/Conv2Dф
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp░
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_35/BiasAdd~
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_35/Relu╩
max_pooling2d_35/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_35/MaxPool┤
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_36/Conv2D/ReadVariableOpП
conv2d_36/Conv2DConv2D!max_pooling2d_35/MaxPool:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_36/Conv2DФ
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp▒
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_36/BiasAdd
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_36/Relu╦
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPoolх
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_37/Conv2D/ReadVariableOpП
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_37/Conv2DФ
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp▒
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_37/Relu╦
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPoolх
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_38/Conv2D/ReadVariableOpП
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
conv2d_38/Conv2DФ
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
conv2d_38/Relu╦
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolх
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_39/Conv2D/ReadVariableOpП
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_39/Conv2DФ
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▒
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_39/Relu╦
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPooly
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_21/dropout/ConstИ
dropout_21/dropout/MulMul!max_pooling2d_39/MaxPool:output:0!dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout_21/dropout/MulЁ
dropout_21/dropout/ShapeShape!max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shapeя
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformІ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_21/dropout/GreaterEqual/yз
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2!
dropout_21/dropout/GreaterEqualЕ
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout_21/dropout/Cast»
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout_21/dropout/Mul_1s
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_7/Constю
flatten_7/ReshapeReshapedropout_21/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2
flatten_7/Reshapeф
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02 
dense_21/MatMul/ReadVariableOpБ
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_21/MatMulе
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_21/BiasAdd/ReadVariableOpд
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_21/BiasAddt
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_21/Reluy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_22/dropout/Constф
dropout_22/dropout/MulMuldense_21/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_22/dropout/Mul
dropout_22/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shapeо
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_22/dropout/random_uniform/RandomUniformІ
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_22/dropout/GreaterEqual/yв
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_22/dropout/GreaterEqualА
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_22/dropout/CastД
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_22/dropout/Mul_1ф
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_22/MatMul/ReadVariableOpЦ
dense_22/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_22/MatMulе
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_22/BiasAdd/ReadVariableOpд
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_22/BiasAddt
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_22/Reluy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_23/dropout/Constф
dropout_23/dropout/MulMuldense_22/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_23/dropout/Mul
dropout_23/dropout/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shapeо
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_23/dropout/random_uniform/RandomUniformІ
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_23/dropout/GreaterEqual/yв
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_23/dropout/GreaterEqualА
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_23/dropout/CastД
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_23/dropout/Mul_1┘
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulл
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulл
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul 
IdentityIdentitydropout_23/dropout/Mul_1:z:0%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ь
Л
6__inference_batch_normalization_7_layer_call_fn_937508

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9346562
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Фђ
┐
__inference__traced_save_938138
file_prefix.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop/
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop/
+savev2_conv2d_39_kernel_read_readvariableop-
)savev2_conv2d_39_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop6
2savev2_adam_conv2d_35_kernel_m_read_readvariableop4
0savev2_adam_conv2d_35_bias_m_read_readvariableop6
2savev2_adam_conv2d_36_kernel_m_read_readvariableop4
0savev2_adam_conv2d_36_bias_m_read_readvariableop6
2savev2_adam_conv2d_37_kernel_m_read_readvariableop4
0savev2_adam_conv2d_37_bias_m_read_readvariableop6
2savev2_adam_conv2d_38_kernel_m_read_readvariableop4
0savev2_adam_conv2d_38_bias_m_read_readvariableop6
2savev2_adam_conv2d_39_kernel_m_read_readvariableop4
0savev2_adam_conv2d_39_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop6
2savev2_adam_conv2d_35_kernel_v_read_readvariableop4
0savev2_adam_conv2d_35_bias_v_read_readvariableop6
2savev2_adam_conv2d_36_kernel_v_read_readvariableop4
0savev2_adam_conv2d_36_bias_v_read_readvariableop6
2savev2_adam_conv2d_37_kernel_v_read_readvariableop4
0savev2_adam_conv2d_37_bias_v_read_readvariableop6
2savev2_adam_conv2d_38_kernel_v_read_readvariableop4
0savev2_adam_conv2d_38_bias_v_read_readvariableop6
2savev2_adam_conv2d_39_kernel_v_read_readvariableop4
0savev2_adam_conv2d_39_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameї"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*ъ!
valueћ!BЉ!BB)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЈ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*Ў
valueЈBїBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╩
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop2savev2_adam_conv2d_35_kernel_m_read_readvariableop0savev2_adam_conv2d_35_bias_m_read_readvariableop2savev2_adam_conv2d_36_kernel_m_read_readvariableop0savev2_adam_conv2d_36_bias_m_read_readvariableop2savev2_adam_conv2d_37_kernel_m_read_readvariableop0savev2_adam_conv2d_37_bias_m_read_readvariableop2savev2_adam_conv2d_38_kernel_m_read_readvariableop0savev2_adam_conv2d_38_bias_m_read_readvariableop2savev2_adam_conv2d_39_kernel_m_read_readvariableop0savev2_adam_conv2d_39_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop2savev2_adam_conv2d_35_kernel_v_read_readvariableop0savev2_adam_conv2d_35_bias_v_read_readvariableop2savev2_adam_conv2d_36_kernel_v_read_readvariableop0savev2_adam_conv2d_36_bias_v_read_readvariableop2savev2_adam_conv2d_37_kernel_v_read_readvariableop0savev2_adam_conv2d_37_bias_v_read_readvariableop2savev2_adam_conv2d_38_kernel_v_read_readvariableop0savev2_adam_conv2d_38_bias_v_read_readvariableop2savev2_adam_conv2d_39_kernel_v_read_readvariableop0savev2_adam_conv2d_39_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *P
dtypesF
D2B	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*Ѕ
_input_shapesэ
З: :	ђ:: : : : : ::: : : ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:
ђ ђ:ђ:
ђђ:ђ::: : : : :	ђ:::: : : ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:
ђ ђ:ђ:
ђђ:ђ:	ђ:::: : : ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:
ђ ђ:ђ:
ђђ:ђ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ђ: 
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
: ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђ ђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ: 

_output_shapes
:: 

_output_shapes
::
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
:	ђ: 
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
: ђ:!%

_output_shapes	
:ђ:.&*
(
_output_shapes
:ђђ:!'

_output_shapes	
:ђ:.(*
(
_output_shapes
:ђђ:!)

_output_shapes	
:ђ:.**
(
_output_shapes
:ђђ:!+

_output_shapes	
:ђ:&,"
 
_output_shapes
:
ђ ђ:!-

_output_shapes	
:ђ:&."
 
_output_shapes
:
ђђ:!/

_output_shapes	
:ђ:%0!

_output_shapes
:	ђ: 1
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
: ђ:!7

_output_shapes	
:ђ:.8*
(
_output_shapes
:ђђ:!9

_output_shapes	
:ђ:.:*
(
_output_shapes
:ђђ:!;

_output_shapes	
:ђ:.<*
(
_output_shapes
:ђђ:!=

_output_shapes	
:ђ:&>"
 
_output_shapes
:
ђ ђ:!?

_output_shapes	
:ђ:&@"
 
_output_shapes
:
ђђ:!A

_output_shapes	
:ђ:B

_output_shapes
: 
э
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937619

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ў
ѓ
-__inference_sequential_7_layer_call_fn_936929

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ
identityѕбStatefulPartitionedCallН
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
:         ђ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_9350542
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ж
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_937769

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
А
Ђ
E__inference_conv2d_37_layer_call_and_return_conditional_losses_934917

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ю
ђ
E__inference_conv2d_36_layer_call_and_return_conditional_losses_934899

inputs9
conv2d_readvariableop_resource: ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         %% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         %% 
 
_user_specified_nameinputs
┐
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937583

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┐
│
E__inference_conv2d_35_layer_call_and_return_conditional_losses_934881

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu¤
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
э
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_937816

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ш
d
+__inference_dropout_21_layer_call_fn_937741

inputs
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_9351852
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
У
│
__inference_loss_fn_2_937920N
:dense_22_kernel_regularizer_square_readvariableop_resource:
ђђ
identityѕб1dense_22/kernel/Regularizer/Square/ReadVariableOpс
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_22_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulџ
IdentityIdentity#dense_22/kernel/Regularizer/mul:z:02^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp
р
E
)__inference_lambda_7_layer_call_fn_937474

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_lambda_7_layer_call_and_return_conditional_losses_9348352
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╔
G
+__inference_dropout_23_layer_call_fn_937865

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_9350332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
э
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_935003

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Д
Ў
)__inference_dense_21_layer_call_fn_937784

inputs
unknown:
ђ ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_9349922
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ 
 
_user_specified_nameinputs
┬
`
D__inference_lambda_7_layer_call_and_return_conditional_losses_935298

inputs
identityЃ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЄ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1Є
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2§
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
с
┤
%__inference_CNN3_layer_call_fn_936228
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:
identityѕбStatefulPartitionedCallж
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
:         *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_CNN3_layer_call_and_return_conditional_losses_9357322
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
╣
г
D__inference_dense_21_layer_call_and_return_conditional_losses_937801

inputs2
matmul_readvariableop_resource:
ђ ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
ReluК
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ 
 
_user_specified_nameinputs
╠▓
Ф
@__inference_CNN3_layer_call_and_return_conditional_losses_936702
input_1H
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_35_conv2d_readvariableop_resource: D
6sequential_7_conv2d_35_biasadd_readvariableop_resource: P
5sequential_7_conv2d_36_conv2d_readvariableop_resource: ђE
6sequential_7_conv2d_36_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_37_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_37_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_38_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_38_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_39_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_39_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_21_matmul_readvariableop_resource:
ђ ђD
5sequential_7_dense_21_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_22_matmul_readvariableop_resource:
ђђD
5sequential_7_dense_22_biasadd_readvariableop_resource:	ђ:
'dense_23_matmul_readvariableop_resource:	ђ6
(dense_23_biasadd_readvariableop_resource:
identityѕб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpбBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_7/batch_normalization_7/ReadVariableOpб3sequential_7/batch_normalization_7/ReadVariableOp_1б-sequential_7/conv2d_35/BiasAdd/ReadVariableOpб,sequential_7/conv2d_35/Conv2D/ReadVariableOpб-sequential_7/conv2d_36/BiasAdd/ReadVariableOpб,sequential_7/conv2d_36/Conv2D/ReadVariableOpб-sequential_7/conv2d_37/BiasAdd/ReadVariableOpб,sequential_7/conv2d_37/Conv2D/ReadVariableOpб-sequential_7/conv2d_38/BiasAdd/ReadVariableOpб,sequential_7/conv2d_38/Conv2D/ReadVariableOpб-sequential_7/conv2d_39/BiasAdd/ReadVariableOpб,sequential_7/conv2d_39/Conv2D/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOp»
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2В
#sequential_7/lambda_7/strided_sliceStridedSliceinput_12sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_sliceП
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpс
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1љ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpќ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_35/Conv2D/ReadVariableOpЎ
sequential_7/conv2d_35/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_35/Conv2DЛ
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpС
sequential_7/conv2d_35/BiasAddBiasAdd&sequential_7/conv2d_35/Conv2D:output:05sequential_7/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_35/BiasAddЦ
sequential_7/conv2d_35/ReluRelu'sequential_7/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_35/Reluы
%sequential_7/max_pooling2d_35/MaxPoolMaxPool)sequential_7/conv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_35/MaxPool█
,sequential_7/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_7/conv2d_36/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_36/Conv2DConv2D.sequential_7/max_pooling2d_35/MaxPool:output:04sequential_7/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_7/conv2d_36/Conv2Dм
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpт
sequential_7/conv2d_36/BiasAddBiasAdd&sequential_7/conv2d_36/Conv2D:output:05sequential_7/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_7/conv2d_36/BiasAddд
sequential_7/conv2d_36/ReluRelu'sequential_7/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_7/conv2d_36/ReluЫ
%sequential_7/max_pooling2d_36/MaxPoolMaxPool)sequential_7/conv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_36/MaxPool▄
,sequential_7/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_37/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_37/Conv2DConv2D.sequential_7/max_pooling2d_36/MaxPool:output:04sequential_7/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_37/Conv2Dм
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpт
sequential_7/conv2d_37/BiasAddBiasAdd&sequential_7/conv2d_37/Conv2D:output:05sequential_7/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_37/BiasAddд
sequential_7/conv2d_37/ReluRelu'sequential_7/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_37/ReluЫ
%sequential_7/max_pooling2d_37/MaxPoolMaxPool)sequential_7/conv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_37/MaxPool▄
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_38/Conv2DConv2D.sequential_7/max_pooling2d_37/MaxPool:output:04sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2Dм
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpт
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_7/conv2d_38/BiasAddд
sequential_7/conv2d_38/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_7/conv2d_38/ReluЫ
%sequential_7/max_pooling2d_38/MaxPoolMaxPool)sequential_7/conv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPool▄
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2Dм
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpт
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_39/BiasAddд
sequential_7/conv2d_39/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_39/ReluЫ
%sequential_7/max_pooling2d_39/MaxPoolMaxPool)sequential_7/conv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPool╗
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2"
 sequential_7/dropout_21/IdentityЇ
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Constл
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2 
sequential_7/flatten_7/ReshapeЛ
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOpО
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/MatMul¤
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/BiasAddЏ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/ReluГ
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_7/dropout_22/IdentityЛ
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/MatMul¤
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/BiasAddЏ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/ReluГ
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_7/dropout_23/IdentityЕ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЦ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/SoftmaxТ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulП
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulП
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul┘	
IdentityIdentitydense_23/Softmax:softmax:03^conv2d_35/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_35/BiasAdd/ReadVariableOp-^sequential_7/conv2d_35/Conv2D/ReadVariableOp.^sequential_7/conv2d_36/BiasAdd/ReadVariableOp-^sequential_7/conv2d_36/Conv2D/ReadVariableOp.^sequential_7/conv2d_37/BiasAdd/ReadVariableOp-^sequential_7/conv2d_37/Conv2D/ReadVariableOp.^sequential_7/conv2d_38/BiasAdd/ReadVariableOp-^sequential_7/conv2d_38/Conv2D/ReadVariableOp.^sequential_7/conv2d_39/BiasAdd/ReadVariableOp-^sequential_7/conv2d_39/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2ѕ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp-sequential_7/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_35/Conv2D/ReadVariableOp,sequential_7/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_36/BiasAdd/ReadVariableOp-sequential_7/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_36/Conv2D/ReadVariableOp,sequential_7/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_37/BiasAdd/ReadVariableOp-sequential_7/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_37/Conv2D/ReadVariableOp,sequential_7/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_38/BiasAdd/ReadVariableOp-sequential_7/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_38/Conv2D/ReadVariableOp,sequential_7/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_39/BiasAdd/ReadVariableOp-sequential_7/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_39/Conv2D/ReadVariableOp,sequential_7/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
▒Њ
С
__inference_call_887269

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_35_conv2d_readvariableop_resource: D
6sequential_7_conv2d_35_biasadd_readvariableop_resource: P
5sequential_7_conv2d_36_conv2d_readvariableop_resource: ђE
6sequential_7_conv2d_36_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_37_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_37_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_38_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_38_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_39_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_39_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_21_matmul_readvariableop_resource:
ђ ђD
5sequential_7_dense_21_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_22_matmul_readvariableop_resource:
ђђD
5sequential_7_dense_22_biasadd_readvariableop_resource:	ђ:
'dense_23_matmul_readvariableop_resource:	ђ6
(dense_23_biasadd_readvariableop_resource:
identityѕбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpбBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_7/batch_normalization_7/ReadVariableOpб3sequential_7/batch_normalization_7/ReadVariableOp_1б-sequential_7/conv2d_35/BiasAdd/ReadVariableOpб,sequential_7/conv2d_35/Conv2D/ReadVariableOpб-sequential_7/conv2d_36/BiasAdd/ReadVariableOpб,sequential_7/conv2d_36/Conv2D/ReadVariableOpб-sequential_7/conv2d_37/BiasAdd/ReadVariableOpб,sequential_7/conv2d_37/Conv2D/ReadVariableOpб-sequential_7/conv2d_38/BiasAdd/ReadVariableOpб,sequential_7/conv2d_38/Conv2D/ReadVariableOpб-sequential_7/conv2d_39/BiasAdd/ReadVariableOpб,sequential_7/conv2d_39/Conv2D/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOp»
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2в
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_sliceП
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpс
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1љ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpќ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_35/Conv2D/ReadVariableOpЎ
sequential_7/conv2d_35/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_35/Conv2DЛ
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpС
sequential_7/conv2d_35/BiasAddBiasAdd&sequential_7/conv2d_35/Conv2D:output:05sequential_7/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_35/BiasAddЦ
sequential_7/conv2d_35/ReluRelu'sequential_7/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_35/Reluы
%sequential_7/max_pooling2d_35/MaxPoolMaxPool)sequential_7/conv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_35/MaxPool█
,sequential_7/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_7/conv2d_36/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_36/Conv2DConv2D.sequential_7/max_pooling2d_35/MaxPool:output:04sequential_7/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_7/conv2d_36/Conv2Dм
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpт
sequential_7/conv2d_36/BiasAddBiasAdd&sequential_7/conv2d_36/Conv2D:output:05sequential_7/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_7/conv2d_36/BiasAddд
sequential_7/conv2d_36/ReluRelu'sequential_7/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_7/conv2d_36/ReluЫ
%sequential_7/max_pooling2d_36/MaxPoolMaxPool)sequential_7/conv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_36/MaxPool▄
,sequential_7/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_37/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_37/Conv2DConv2D.sequential_7/max_pooling2d_36/MaxPool:output:04sequential_7/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_37/Conv2Dм
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpт
sequential_7/conv2d_37/BiasAddBiasAdd&sequential_7/conv2d_37/Conv2D:output:05sequential_7/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_37/BiasAddд
sequential_7/conv2d_37/ReluRelu'sequential_7/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_37/ReluЫ
%sequential_7/max_pooling2d_37/MaxPoolMaxPool)sequential_7/conv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_37/MaxPool▄
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_38/Conv2DConv2D.sequential_7/max_pooling2d_37/MaxPool:output:04sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2Dм
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpт
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_7/conv2d_38/BiasAddд
sequential_7/conv2d_38/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_7/conv2d_38/ReluЫ
%sequential_7/max_pooling2d_38/MaxPoolMaxPool)sequential_7/conv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPool▄
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2Dм
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpт
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_39/BiasAddд
sequential_7/conv2d_39/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_39/ReluЫ
%sequential_7/max_pooling2d_39/MaxPoolMaxPool)sequential_7/conv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPool╗
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2"
 sequential_7/dropout_21/IdentityЇ
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Constл
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2 
sequential_7/flatten_7/ReshapeЛ
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOpО
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/MatMul¤
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/BiasAddЏ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/ReluГ
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_7/dropout_22/IdentityЛ
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/MatMul¤
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/BiasAddЏ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/ReluГ
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_7/dropout_23/IdentityЕ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЦ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/Softmax╝
IdentityIdentitydense_23/Softmax:softmax:0 ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_35/BiasAdd/ReadVariableOp-^sequential_7/conv2d_35/Conv2D/ReadVariableOp.^sequential_7/conv2d_36/BiasAdd/ReadVariableOp-^sequential_7/conv2d_36/Conv2D/ReadVariableOp.^sequential_7/conv2d_37/BiasAdd/ReadVariableOp-^sequential_7/conv2d_37/Conv2D/ReadVariableOp.^sequential_7/conv2d_38/BiasAdd/ReadVariableOp-^sequential_7/conv2d_38/Conv2D/ReadVariableOp.^sequential_7/conv2d_39/BiasAdd/ReadVariableOp-^sequential_7/conv2d_39/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2ѕ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp-sequential_7/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_35/Conv2D/ReadVariableOp,sequential_7/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_36/BiasAdd/ReadVariableOp-sequential_7/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_36/Conv2D/ReadVariableOp,sequential_7/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_37/BiasAdd/ReadVariableOp-sequential_7/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_37/Conv2D/ReadVariableOp,sequential_7/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_38/BiasAdd/ReadVariableOp-sequential_7/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_38/Conv2D/ReadVariableOp,sequential_7/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_39/BiasAdd/ReadVariableOp-sequential_7/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_39/Conv2D/ReadVariableOp,sequential_7/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┐
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_934700

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
э
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_935033

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╬
А
*__inference_conv2d_36_layer_call_fn_937660

inputs"
unknown: ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_36_layer_call_and_return_conditional_losses_9348992
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         %%ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         %% : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         %% 
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_934814

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
В
Л
6__inference_batch_normalization_7_layer_call_fn_937521

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9347002
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
Ъ
*__inference_conv2d_35_layer_call_fn_937634

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_9348812
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ЂЉ
С
__inference_call_889595

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_35_conv2d_readvariableop_resource: D
6sequential_7_conv2d_35_biasadd_readvariableop_resource: P
5sequential_7_conv2d_36_conv2d_readvariableop_resource: ђE
6sequential_7_conv2d_36_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_37_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_37_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_38_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_38_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_39_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_39_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_21_matmul_readvariableop_resource:
ђ ђD
5sequential_7_dense_21_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_22_matmul_readvariableop_resource:
ђђD
5sequential_7_dense_22_biasadd_readvariableop_resource:	ђ:
'dense_23_matmul_readvariableop_resource:	ђ6
(dense_23_biasadd_readvariableop_resource:
identityѕбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpбBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_7/batch_normalization_7/ReadVariableOpб3sequential_7/batch_normalization_7/ReadVariableOp_1б-sequential_7/conv2d_35/BiasAdd/ReadVariableOpб,sequential_7/conv2d_35/Conv2D/ReadVariableOpб-sequential_7/conv2d_36/BiasAdd/ReadVariableOpб,sequential_7/conv2d_36/Conv2D/ReadVariableOpб-sequential_7/conv2d_37/BiasAdd/ReadVariableOpб,sequential_7/conv2d_37/Conv2D/ReadVariableOpб-sequential_7/conv2d_38/BiasAdd/ReadVariableOpб,sequential_7/conv2d_38/Conv2D/ReadVariableOpб-sequential_7/conv2d_39/BiasAdd/ReadVariableOpб,sequential_7/conv2d_39/Conv2D/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOp»
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2с
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ђKK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_sliceП
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpс
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1љ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpќ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1║
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:ђKK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_35/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_35/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK *
paddingSAME*
strides
2
sequential_7/conv2d_35/Conv2DЛ
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp▄
sequential_7/conv2d_35/BiasAddBiasAdd&sequential_7/conv2d_35/Conv2D:output:05sequential_7/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK 2 
sequential_7/conv2d_35/BiasAddЮ
sequential_7/conv2d_35/ReluRelu'sequential_7/conv2d_35/BiasAdd:output:0*
T0*'
_output_shapes
:ђKK 2
sequential_7/conv2d_35/Reluж
%sequential_7/max_pooling2d_35/MaxPoolMaxPool)sequential_7/conv2d_35/Relu:activations:0*'
_output_shapes
:ђ%% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_35/MaxPool█
,sequential_7/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_7/conv2d_36/Conv2D/ReadVariableOpЅ
sequential_7/conv2d_36/Conv2DConv2D.sequential_7/max_pooling2d_35/MaxPool:output:04sequential_7/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ*
paddingSAME*
strides
2
sequential_7/conv2d_36/Conv2Dм
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpП
sequential_7/conv2d_36/BiasAddBiasAdd&sequential_7/conv2d_36/Conv2D:output:05sequential_7/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ2 
sequential_7/conv2d_36/BiasAddъ
sequential_7/conv2d_36/ReluRelu'sequential_7/conv2d_36/BiasAdd:output:0*
T0*(
_output_shapes
:ђ%%ђ2
sequential_7/conv2d_36/ReluЖ
%sequential_7/max_pooling2d_36/MaxPoolMaxPool)sequential_7/conv2d_36/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_36/MaxPool▄
,sequential_7/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_37/Conv2D/ReadVariableOpЅ
sequential_7/conv2d_37/Conv2DConv2D.sequential_7/max_pooling2d_36/MaxPool:output:04sequential_7/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_7/conv2d_37/Conv2Dм
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpП
sequential_7/conv2d_37/BiasAddBiasAdd&sequential_7/conv2d_37/Conv2D:output:05sequential_7/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_7/conv2d_37/BiasAddъ
sequential_7/conv2d_37/ReluRelu'sequential_7/conv2d_37/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_7/conv2d_37/ReluЖ
%sequential_7/max_pooling2d_37/MaxPoolMaxPool)sequential_7/conv2d_37/Relu:activations:0*(
_output_shapes
:ђ		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_37/MaxPool▄
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpЅ
sequential_7/conv2d_38/Conv2DConv2D.sequential_7/max_pooling2d_37/MaxPool:output:04sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ		ђ*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2Dм
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpП
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ		ђ2 
sequential_7/conv2d_38/BiasAddъ
sequential_7/conv2d_38/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*(
_output_shapes
:ђ		ђ2
sequential_7/conv2d_38/ReluЖ
%sequential_7/max_pooling2d_38/MaxPoolMaxPool)sequential_7/conv2d_38/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPool▄
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpЅ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2Dм
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpП
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_7/conv2d_39/BiasAddъ
sequential_7/conv2d_39/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_7/conv2d_39/ReluЖ
%sequential_7/max_pooling2d_39/MaxPoolMaxPool)sequential_7/conv2d_39/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPool│
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_39/MaxPool:output:0*
T0*(
_output_shapes
:ђђ2"
 sequential_7/dropout_21/IdentityЇ
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Const╚
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0* 
_output_shapes
:
ђђ 2 
sequential_7/flatten_7/ReshapeЛ
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOp¤
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_21/MatMul¤
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOpм
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_21/BiasAddЊ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_21/ReluЦ
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_7/dropout_22/IdentityЛ
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOpЛ
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_22/MatMul¤
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOpм
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_22/BiasAddЊ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_7/dense_22/ReluЦ
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_7/dropout_23/IdentityЕ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_23/MatMul/ReadVariableOpЕ
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЮ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_23/BiasAddt
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*
_output_shapes
:	ђ2
dense_23/Softmax┤
IdentityIdentitydense_23/Softmax:softmax:0 ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_35/BiasAdd/ReadVariableOp-^sequential_7/conv2d_35/Conv2D/ReadVariableOp.^sequential_7/conv2d_36/BiasAdd/ReadVariableOp-^sequential_7/conv2d_36/Conv2D/ReadVariableOp.^sequential_7/conv2d_37/BiasAdd/ReadVariableOp-^sequential_7/conv2d_37/Conv2D/ReadVariableOp.^sequential_7/conv2d_38/BiasAdd/ReadVariableOp-^sequential_7/conv2d_38/Conv2D/ReadVariableOp.^sequential_7/conv2d_39/BiasAdd/ReadVariableOp-^sequential_7/conv2d_39/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*
_output_shapes
:	ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ђKK: : : : : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2ѕ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp-sequential_7/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_35/Conv2D/ReadVariableOp,sequential_7/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_36/BiasAdd/ReadVariableOp-sequential_7/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_36/Conv2D/ReadVariableOp,sequential_7/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_37/BiasAdd/ReadVariableOp-sequential_7/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_37/Conv2D/ReadVariableOp,sequential_7/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_38/BiasAdd/ReadVariableOp-sequential_7/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_38/Conv2D/ReadVariableOp,sequential_7/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_39/BiasAdd/ReadVariableOp-sequential_7/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_39/Conv2D/ReadVariableOp,sequential_7/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:O K
'
_output_shapes
:ђKK
 
_user_specified_nameinputs
І
ю
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937565

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ќ
d
F__inference_dropout_21_layer_call_and_return_conditional_losses_934965

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_35_layer_call_fn_934772

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_9347662
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Л
б
*__inference_conv2d_38_layer_call_fn_937700

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_38_layer_call_and_return_conditional_losses_9349352
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
И

Ш
D__inference_dense_23_layer_call_and_return_conditional_losses_935707

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Я
│
%__inference_CNN3_layer_call_fn_936273

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:
identityѕбStatefulPartitionedCallУ
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
:         *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_CNN3_layer_call_and_return_conditional_losses_9357322
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Л
б
*__inference_conv2d_37_layer_call_fn_937680

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_37_layer_call_and_return_conditional_losses_9349172
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
х
e
F__inference_dropout_23_layer_call_and_return_conditional_losses_937887

inputs
identityѕc
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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ц
Л
6__inference_batch_normalization_7_layer_call_fn_937547

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9352712
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
├
ю
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937601

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┼▀
Ћ
@__inference_CNN3_layer_call_and_return_conditional_losses_936829
input_1H
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_35_conv2d_readvariableop_resource: D
6sequential_7_conv2d_35_biasadd_readvariableop_resource: P
5sequential_7_conv2d_36_conv2d_readvariableop_resource: ђE
6sequential_7_conv2d_36_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_37_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_37_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_38_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_38_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_39_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_39_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_21_matmul_readvariableop_resource:
ђ ђD
5sequential_7_dense_21_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_22_matmul_readvariableop_resource:
ђђD
5sequential_7_dense_22_biasadd_readvariableop_resource:	ђ:
'dense_23_matmul_readvariableop_resource:	ђ6
(dense_23_biasadd_readvariableop_resource:
identityѕб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpб1sequential_7/batch_normalization_7/AssignNewValueб3sequential_7/batch_normalization_7/AssignNewValue_1бBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_7/batch_normalization_7/ReadVariableOpб3sequential_7/batch_normalization_7/ReadVariableOp_1б-sequential_7/conv2d_35/BiasAdd/ReadVariableOpб,sequential_7/conv2d_35/Conv2D/ReadVariableOpб-sequential_7/conv2d_36/BiasAdd/ReadVariableOpб,sequential_7/conv2d_36/Conv2D/ReadVariableOpб-sequential_7/conv2d_37/BiasAdd/ReadVariableOpб,sequential_7/conv2d_37/Conv2D/ReadVariableOpб-sequential_7/conv2d_38/BiasAdd/ReadVariableOpб,sequential_7/conv2d_38/Conv2D/ReadVariableOpб-sequential_7/conv2d_39/BiasAdd/ReadVariableOpб,sequential_7/conv2d_39/Conv2D/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOp»
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2В
#sequential_7/lambda_7/strided_sliceStridedSliceinput_12sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_sliceП
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpс
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1љ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpќ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1л
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<25
3sequential_7/batch_normalization_7/FusedBatchNormV3ы
1sequential_7/batch_normalization_7/AssignNewValueAssignVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource@sequential_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_7/batch_normalization_7/AssignNewValue§
3sequential_7/batch_normalization_7/AssignNewValue_1AssignVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceDsequential_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0E^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_7/batch_normalization_7/AssignNewValue_1┌
,sequential_7/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_35/Conv2D/ReadVariableOpЎ
sequential_7/conv2d_35/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_35/Conv2DЛ
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpС
sequential_7/conv2d_35/BiasAddBiasAdd&sequential_7/conv2d_35/Conv2D:output:05sequential_7/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_35/BiasAddЦ
sequential_7/conv2d_35/ReluRelu'sequential_7/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_35/Reluы
%sequential_7/max_pooling2d_35/MaxPoolMaxPool)sequential_7/conv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_35/MaxPool█
,sequential_7/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_7/conv2d_36/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_36/Conv2DConv2D.sequential_7/max_pooling2d_35/MaxPool:output:04sequential_7/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_7/conv2d_36/Conv2Dм
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpт
sequential_7/conv2d_36/BiasAddBiasAdd&sequential_7/conv2d_36/Conv2D:output:05sequential_7/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_7/conv2d_36/BiasAddд
sequential_7/conv2d_36/ReluRelu'sequential_7/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_7/conv2d_36/ReluЫ
%sequential_7/max_pooling2d_36/MaxPoolMaxPool)sequential_7/conv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_36/MaxPool▄
,sequential_7/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_37/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_37/Conv2DConv2D.sequential_7/max_pooling2d_36/MaxPool:output:04sequential_7/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_37/Conv2Dм
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpт
sequential_7/conv2d_37/BiasAddBiasAdd&sequential_7/conv2d_37/Conv2D:output:05sequential_7/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_37/BiasAddд
sequential_7/conv2d_37/ReluRelu'sequential_7/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_37/ReluЫ
%sequential_7/max_pooling2d_37/MaxPoolMaxPool)sequential_7/conv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_37/MaxPool▄
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_38/Conv2DConv2D.sequential_7/max_pooling2d_37/MaxPool:output:04sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2Dм
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpт
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_7/conv2d_38/BiasAddд
sequential_7/conv2d_38/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_7/conv2d_38/ReluЫ
%sequential_7/max_pooling2d_38/MaxPoolMaxPool)sequential_7/conv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPool▄
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2Dм
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpт
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_39/BiasAddд
sequential_7/conv2d_39/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_39/ReluЫ
%sequential_7/max_pooling2d_39/MaxPoolMaxPool)sequential_7/conv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPoolЊ
%sequential_7/dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2'
%sequential_7/dropout_21/dropout/ConstВ
#sequential_7/dropout_21/dropout/MulMul.sequential_7/max_pooling2d_39/MaxPool:output:0.sequential_7/dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2%
#sequential_7/dropout_21/dropout/Mulг
%sequential_7/dropout_21/dropout/ShapeShape.sequential_7/max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_21/dropout/ShapeЁ
<sequential_7/dropout_21/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02>
<sequential_7/dropout_21/dropout/random_uniform/RandomUniformЦ
.sequential_7/dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_7/dropout_21/dropout/GreaterEqual/yД
,sequential_7/dropout_21/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_21/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2.
,sequential_7/dropout_21/dropout/GreaterEqualл
$sequential_7/dropout_21/dropout/CastCast0sequential_7/dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2&
$sequential_7/dropout_21/dropout/Castс
%sequential_7/dropout_21/dropout/Mul_1Mul'sequential_7/dropout_21/dropout/Mul:z:0(sequential_7/dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2'
%sequential_7/dropout_21/dropout/Mul_1Ї
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Constл
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/dropout/Mul_1:z:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2 
sequential_7/flatten_7/ReshapeЛ
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOpО
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/MatMul¤
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/BiasAddЏ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/ReluЊ
%sequential_7/dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_22/dropout/Constя
#sequential_7/dropout_22/dropout/MulMul(sequential_7/dense_21/Relu:activations:0.sequential_7/dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_7/dropout_22/dropout/Mulд
%sequential_7/dropout_22/dropout/ShapeShape(sequential_7/dense_21/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_22/dropout/Shape§
<sequential_7/dropout_22/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_7/dropout_22/dropout/random_uniform/RandomUniformЦ
.sequential_7/dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_22/dropout/GreaterEqual/yЪ
,sequential_7/dropout_22/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_22/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_7/dropout_22/dropout/GreaterEqual╚
$sequential_7/dropout_22/dropout/CastCast0sequential_7/dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_7/dropout_22/dropout/Cast█
%sequential_7/dropout_22/dropout/Mul_1Mul'sequential_7/dropout_22/dropout/Mul:z:0(sequential_7/dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_7/dropout_22/dropout/Mul_1Л
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/dropout/Mul_1:z:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/MatMul¤
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/BiasAddЏ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/ReluЊ
%sequential_7/dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_23/dropout/Constя
#sequential_7/dropout_23/dropout/MulMul(sequential_7/dense_22/Relu:activations:0.sequential_7/dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_7/dropout_23/dropout/Mulд
%sequential_7/dropout_23/dropout/ShapeShape(sequential_7/dense_22/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_23/dropout/Shape§
<sequential_7/dropout_23/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_7/dropout_23/dropout/random_uniform/RandomUniformЦ
.sequential_7/dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_23/dropout/GreaterEqual/yЪ
,sequential_7/dropout_23/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_23/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_7/dropout_23/dropout/GreaterEqual╚
$sequential_7/dropout_23/dropout/CastCast0sequential_7/dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_7/dropout_23/dropout/Cast█
%sequential_7/dropout_23/dropout/Mul_1Mul'sequential_7/dropout_23/dropout/Mul:z:0(sequential_7/dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_7/dropout_23/dropout/Mul_1Е
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЦ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/SoftmaxТ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulП
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulП
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul├

IdentityIdentitydense_23/Softmax:softmax:03^conv2d_35/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp2^sequential_7/batch_normalization_7/AssignNewValue4^sequential_7/batch_normalization_7/AssignNewValue_1C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_35/BiasAdd/ReadVariableOp-^sequential_7/conv2d_35/Conv2D/ReadVariableOp.^sequential_7/conv2d_36/BiasAdd/ReadVariableOp-^sequential_7/conv2d_36/Conv2D/ReadVariableOp.^sequential_7/conv2d_37/BiasAdd/ReadVariableOp-^sequential_7/conv2d_37/Conv2D/ReadVariableOp.^sequential_7/conv2d_38/BiasAdd/ReadVariableOp-^sequential_7/conv2d_38/Conv2D/ReadVariableOp.^sequential_7/conv2d_39/BiasAdd/ReadVariableOp-^sequential_7/conv2d_39/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2f
1sequential_7/batch_normalization_7/AssignNewValue1sequential_7/batch_normalization_7/AssignNewValue2j
3sequential_7/batch_normalization_7/AssignNewValue_13sequential_7/batch_normalization_7/AssignNewValue_12ѕ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp-sequential_7/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_35/Conv2D/ReadVariableOp,sequential_7/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_36/BiasAdd/ReadVariableOp-sequential_7/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_36/Conv2D/ReadVariableOp,sequential_7/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_37/BiasAdd/ReadVariableOp-sequential_7/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_37/Conv2D/ReadVariableOp,sequential_7/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_38/BiasAdd/ReadVariableOp-sequential_7/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_38/Conv2D/ReadVariableOp,sequential_7/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_39/BiasAdd/ReadVariableOp-sequential_7/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_39/Conv2D/ReadVariableOp,sequential_7/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
А
Ђ
E__inference_conv2d_39_layer_call_and_return_conditional_losses_934953

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
e
F__inference_dropout_21_layer_call_and_return_conditional_losses_937758

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yК
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
├
│
$__inference_signature_wrapper_936183
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:
identityѕбStatefulPartitionedCall╩
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
:         *6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ **
f%R#
!__inference__wrapped_model_9346342
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
яЇ
»
H__inference_sequential_7_layer_call_and_return_conditional_losses_937329
lambda_7_input;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_35_conv2d_readvariableop_resource: 7
)conv2d_35_biasadd_readvariableop_resource: C
(conv2d_36_conv2d_readvariableop_resource: ђ8
)conv2d_36_biasadd_readvariableop_resource:	ђD
(conv2d_37_conv2d_readvariableop_resource:ђђ8
)conv2d_37_biasadd_readvariableop_resource:	ђD
(conv2d_38_conv2d_readvariableop_resource:ђђ8
)conv2d_38_biasadd_readvariableop_resource:	ђD
(conv2d_39_conv2d_readvariableop_resource:ђђ8
)conv2d_39_biasadd_readvariableop_resource:	ђ;
'dense_21_matmul_readvariableop_resource:
ђ ђ7
(dense_21_biasadd_readvariableop_resource:	ђ;
'dense_22_matmul_readvariableop_resource:
ђђ7
(dense_22_biasadd_readvariableop_resource:	ђ
identityѕб5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б conv2d_35/BiasAdd/ReadVariableOpбconv2d_35/Conv2D/ReadVariableOpб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб conv2d_36/BiasAdd/ReadVariableOpбconv2d_36/Conv2D/ReadVariableOpб conv2d_37/BiasAdd/ReadVariableOpбconv2d_37/Conv2D/ReadVariableOpб conv2d_38/BiasAdd/ReadVariableOpбconv2d_38/Conv2D/ReadVariableOpб conv2d_39/BiasAdd/ReadVariableOpбconv2d_39/Conv2D/ReadVariableOpбdense_21/BiasAdd/ReadVariableOpбdense_21/MatMul/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЎ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Ў
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2▓
lambda_7/strided_sliceStridedSlicelambda_7_input%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_7/strided_sliceХ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1ж
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1у
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3│
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_35/Conv2D/ReadVariableOpт
conv2d_35/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_35/Conv2Dф
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp░
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_35/BiasAdd~
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_35/Relu╩
max_pooling2d_35/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_35/MaxPool┤
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_36/Conv2D/ReadVariableOpП
conv2d_36/Conv2DConv2D!max_pooling2d_35/MaxPool:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_36/Conv2DФ
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp▒
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_36/BiasAdd
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_36/Relu╦
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPoolх
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_37/Conv2D/ReadVariableOpП
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_37/Conv2DФ
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp▒
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_37/Relu╦
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPoolх
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_38/Conv2D/ReadVariableOpП
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
conv2d_38/Conv2DФ
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
conv2d_38/Relu╦
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolх
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_39/Conv2D/ReadVariableOpП
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_39/Conv2DФ
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▒
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_39/Relu╦
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPoolћ
dropout_21/IdentityIdentity!max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2
dropout_21/Identitys
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_7/Constю
flatten_7/ReshapeReshapedropout_21/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2
flatten_7/Reshapeф
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02 
dense_21/MatMul/ReadVariableOpБ
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_21/MatMulе
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_21/BiasAdd/ReadVariableOpд
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_21/BiasAddt
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_21/Reluє
dropout_22/IdentityIdentitydense_21/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_22/Identityф
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_22/MatMul/ReadVariableOpЦ
dense_22/MatMulMatMuldropout_22/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_22/MatMulе
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_22/BiasAdd/ReadVariableOpд
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_22/BiasAddt
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_22/Reluє
dropout_23/IdentityIdentitydense_22/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_23/Identity┘
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulл
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulл
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul»
IdentityIdentitydropout_23/Identity:output:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_7_input
Єr
я

H__inference_sequential_7_layer_call_and_return_conditional_losses_935420

inputs*
batch_normalization_7_935348:*
batch_normalization_7_935350:*
batch_normalization_7_935352:*
batch_normalization_7_935354:*
conv2d_35_935357: 
conv2d_35_935359: +
conv2d_36_935363: ђ
conv2d_36_935365:	ђ,
conv2d_37_935369:ђђ
conv2d_37_935371:	ђ,
conv2d_38_935375:ђђ
conv2d_38_935377:	ђ,
conv2d_39_935381:ђђ
conv2d_39_935383:	ђ#
dense_21_935389:
ђ ђ
dense_21_935391:	ђ#
dense_22_935395:
ђђ
dense_22_935397:	ђ
identityѕб-batch_normalization_7/StatefulPartitionedCallб!conv2d_35/StatefulPartitionedCallб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб!conv2d_36/StatefulPartitionedCallб!conv2d_37/StatefulPartitionedCallб!conv2d_38/StatefulPartitionedCallб!conv2d_39/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб1dense_21/kernel/Regularizer/Square/ReadVariableOpб dense_22/StatefulPartitionedCallб1dense_22/kernel/Regularizer/Square/ReadVariableOpб"dropout_21/StatefulPartitionedCallб"dropout_22/StatefulPartitionedCallб"dropout_23/StatefulPartitionedCallр
lambda_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_lambda_7_layer_call_and_return_conditional_losses_9352982
lambda_7/PartitionedCall╗
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0batch_normalization_7_935348batch_normalization_7_935350batch_normalization_7_935352batch_normalization_7_935354*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9352712/
-batch_normalization_7/StatefulPartitionedCallо
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_35_935357conv2d_35_935359*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_9348812#
!conv2d_35/StatefulPartitionedCallЮ
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         %% * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_9347662"
 max_pooling2d_35/PartitionedCall╩
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0conv2d_36_935363conv2d_36_935365*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_36_layer_call_and_return_conditional_losses_9348992#
!conv2d_36/StatefulPartitionedCallъ
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_9347782"
 max_pooling2d_36/PartitionedCall╩
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_37_935369conv2d_37_935371*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_37_layer_call_and_return_conditional_losses_9349172#
!conv2d_37/StatefulPartitionedCallъ
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_9347902"
 max_pooling2d_37/PartitionedCall╩
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_38_935375conv2d_38_935377*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_38_layer_call_and_return_conditional_losses_9349352#
!conv2d_38/StatefulPartitionedCallъ
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_9348022"
 max_pooling2d_38/PartitionedCall╩
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_935381conv2d_39_935383*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_39_layer_call_and_return_conditional_losses_9349532#
!conv2d_39/StatefulPartitionedCallъ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_9348142"
 max_pooling2d_39/PartitionedCallБ
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_9351852$
"dropout_21/StatefulPartitionedCallѓ
flatten_7/PartitionedCallPartitionedCall+dropout_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_9349732
flatten_7/PartitionedCallХ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_21_935389dense_21_935391*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_9349922"
 dense_21/StatefulPartitionedCall└
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_9351462$
"dropout_22/StatefulPartitionedCall┐
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_22_935395dense_22_935397*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_9350222"
 dense_22/StatefulPartitionedCall└
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_9351132$
"dropout_23/StatefulPartitionedCall┴
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_935357*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulИ
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21_935389* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulИ
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_22_935395* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulХ
IdentityIdentity+dropout_23/StatefulPartitionedCall:output:0.^batch_normalization_7/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall2^dense_21/kernel/Regularizer/Square/ReadVariableOp!^dense_22/StatefulPartitionedCall2^dense_22/kernel/Regularizer/Square/ReadVariableOp#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Н
d
+__inference_dropout_22_layer_call_fn_937811

inputs
identityѕбStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_9351462
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
▒Њ
С
__inference_call_889771

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_35_conv2d_readvariableop_resource: D
6sequential_7_conv2d_35_biasadd_readvariableop_resource: P
5sequential_7_conv2d_36_conv2d_readvariableop_resource: ђE
6sequential_7_conv2d_36_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_37_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_37_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_38_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_38_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_39_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_39_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_21_matmul_readvariableop_resource:
ђ ђD
5sequential_7_dense_21_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_22_matmul_readvariableop_resource:
ђђD
5sequential_7_dense_22_biasadd_readvariableop_resource:	ђ:
'dense_23_matmul_readvariableop_resource:	ђ6
(dense_23_biasadd_readvariableop_resource:
identityѕбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpбBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_7/batch_normalization_7/ReadVariableOpб3sequential_7/batch_normalization_7/ReadVariableOp_1б-sequential_7/conv2d_35/BiasAdd/ReadVariableOpб,sequential_7/conv2d_35/Conv2D/ReadVariableOpб-sequential_7/conv2d_36/BiasAdd/ReadVariableOpб,sequential_7/conv2d_36/Conv2D/ReadVariableOpб-sequential_7/conv2d_37/BiasAdd/ReadVariableOpб,sequential_7/conv2d_37/Conv2D/ReadVariableOpб-sequential_7/conv2d_38/BiasAdd/ReadVariableOpб,sequential_7/conv2d_38/Conv2D/ReadVariableOpб-sequential_7/conv2d_39/BiasAdd/ReadVariableOpб,sequential_7/conv2d_39/Conv2D/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOp»
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2в
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_sliceП
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpс
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1љ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpќ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_35/Conv2D/ReadVariableOpЎ
sequential_7/conv2d_35/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_35/Conv2DЛ
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpС
sequential_7/conv2d_35/BiasAddBiasAdd&sequential_7/conv2d_35/Conv2D:output:05sequential_7/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_35/BiasAddЦ
sequential_7/conv2d_35/ReluRelu'sequential_7/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_35/Reluы
%sequential_7/max_pooling2d_35/MaxPoolMaxPool)sequential_7/conv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_35/MaxPool█
,sequential_7/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_7/conv2d_36/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_36/Conv2DConv2D.sequential_7/max_pooling2d_35/MaxPool:output:04sequential_7/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_7/conv2d_36/Conv2Dм
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpт
sequential_7/conv2d_36/BiasAddBiasAdd&sequential_7/conv2d_36/Conv2D:output:05sequential_7/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_7/conv2d_36/BiasAddд
sequential_7/conv2d_36/ReluRelu'sequential_7/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_7/conv2d_36/ReluЫ
%sequential_7/max_pooling2d_36/MaxPoolMaxPool)sequential_7/conv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_36/MaxPool▄
,sequential_7/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_37/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_37/Conv2DConv2D.sequential_7/max_pooling2d_36/MaxPool:output:04sequential_7/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_37/Conv2Dм
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpт
sequential_7/conv2d_37/BiasAddBiasAdd&sequential_7/conv2d_37/Conv2D:output:05sequential_7/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_37/BiasAddд
sequential_7/conv2d_37/ReluRelu'sequential_7/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_37/ReluЫ
%sequential_7/max_pooling2d_37/MaxPoolMaxPool)sequential_7/conv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_37/MaxPool▄
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_38/Conv2DConv2D.sequential_7/max_pooling2d_37/MaxPool:output:04sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2Dм
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpт
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_7/conv2d_38/BiasAddд
sequential_7/conv2d_38/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_7/conv2d_38/ReluЫ
%sequential_7/max_pooling2d_38/MaxPoolMaxPool)sequential_7/conv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPool▄
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2Dм
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpт
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_39/BiasAddд
sequential_7/conv2d_39/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_39/ReluЫ
%sequential_7/max_pooling2d_39/MaxPoolMaxPool)sequential_7/conv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPool╗
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2"
 sequential_7/dropout_21/IdentityЇ
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Constл
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2 
sequential_7/flatten_7/ReshapeЛ
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOpО
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/MatMul¤
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/BiasAddЏ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/ReluГ
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_7/dropout_22/IdentityЛ
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/MatMul¤
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/BiasAddЏ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/ReluГ
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_7/dropout_23/IdentityЕ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЦ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/Softmax╝
IdentityIdentitydense_23/Softmax:softmax:0 ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_35/BiasAdd/ReadVariableOp-^sequential_7/conv2d_35/Conv2D/ReadVariableOp.^sequential_7/conv2d_36/BiasAdd/ReadVariableOp-^sequential_7/conv2d_36/Conv2D/ReadVariableOp.^sequential_7/conv2d_37/BiasAdd/ReadVariableOp-^sequential_7/conv2d_37/Conv2D/ReadVariableOp.^sequential_7/conv2d_38/BiasAdd/ReadVariableOp-^sequential_7/conv2d_38/Conv2D/ReadVariableOp.^sequential_7/conv2d_39/BiasAdd/ReadVariableOp-^sequential_7/conv2d_39/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2ѕ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp-sequential_7/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_35/Conv2D/ReadVariableOp,sequential_7/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_36/BiasAdd/ReadVariableOp-sequential_7/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_36/Conv2D/ReadVariableOp,sequential_7/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_37/BiasAdd/ReadVariableOp-sequential_7/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_37/Conv2D/ReadVariableOp,sequential_7/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_38/BiasAdd/ReadVariableOp-sequential_7/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_38/Conv2D/ReadVariableOp,sequential_7/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_39/BiasAdd/ReadVariableOp-sequential_7/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_39/Conv2D/ReadVariableOp,sequential_7/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Д
Ў
)__inference_dense_22_layer_call_fn_937843

inputs
unknown:
ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_9350222
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╣
г
D__inference_dense_22_layer_call_and_return_conditional_losses_937860

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
ReluК
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
р
E
)__inference_lambda_7_layer_call_fn_937479

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_lambda_7_layer_call_and_return_conditional_losses_9352982
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
О
F
*__inference_flatten_7_layer_call_fn_937763

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_9349732
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
џ
╗
__inference_loss_fn_0_937898U
;conv2d_35_kernel_regularizer_square_readvariableop_resource: 
identityѕб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpВ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_35_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulю
IdentityIdentity$conv2d_35/kernel/Regularizer/mul:z:03^conv2d_35/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp
А
Ђ
E__inference_conv2d_38_layer_call_and_return_conditional_losses_934935

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
┬▀
ћ
@__inference_CNN3_layer_call_and_return_conditional_losses_936596

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_35_conv2d_readvariableop_resource: D
6sequential_7_conv2d_35_biasadd_readvariableop_resource: P
5sequential_7_conv2d_36_conv2d_readvariableop_resource: ђE
6sequential_7_conv2d_36_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_37_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_37_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_38_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_38_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_39_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_39_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_21_matmul_readvariableop_resource:
ђ ђD
5sequential_7_dense_21_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_22_matmul_readvariableop_resource:
ђђD
5sequential_7_dense_22_biasadd_readvariableop_resource:	ђ:
'dense_23_matmul_readvariableop_resource:	ђ6
(dense_23_biasadd_readvariableop_resource:
identityѕб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpб1sequential_7/batch_normalization_7/AssignNewValueб3sequential_7/batch_normalization_7/AssignNewValue_1бBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_7/batch_normalization_7/ReadVariableOpб3sequential_7/batch_normalization_7/ReadVariableOp_1б-sequential_7/conv2d_35/BiasAdd/ReadVariableOpб,sequential_7/conv2d_35/Conv2D/ReadVariableOpб-sequential_7/conv2d_36/BiasAdd/ReadVariableOpб,sequential_7/conv2d_36/Conv2D/ReadVariableOpб-sequential_7/conv2d_37/BiasAdd/ReadVariableOpб,sequential_7/conv2d_37/Conv2D/ReadVariableOpб-sequential_7/conv2d_38/BiasAdd/ReadVariableOpб,sequential_7/conv2d_38/Conv2D/ReadVariableOpб-sequential_7/conv2d_39/BiasAdd/ReadVariableOpб,sequential_7/conv2d_39/Conv2D/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOp»
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2в
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_sliceП
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpс
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1љ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpќ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1л
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<25
3sequential_7/batch_normalization_7/FusedBatchNormV3ы
1sequential_7/batch_normalization_7/AssignNewValueAssignVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource@sequential_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_7/batch_normalization_7/AssignNewValue§
3sequential_7/batch_normalization_7/AssignNewValue_1AssignVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceDsequential_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0E^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_7/batch_normalization_7/AssignNewValue_1┌
,sequential_7/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_35/Conv2D/ReadVariableOpЎ
sequential_7/conv2d_35/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_35/Conv2DЛ
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpС
sequential_7/conv2d_35/BiasAddBiasAdd&sequential_7/conv2d_35/Conv2D:output:05sequential_7/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_35/BiasAddЦ
sequential_7/conv2d_35/ReluRelu'sequential_7/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_35/Reluы
%sequential_7/max_pooling2d_35/MaxPoolMaxPool)sequential_7/conv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_35/MaxPool█
,sequential_7/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_7/conv2d_36/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_36/Conv2DConv2D.sequential_7/max_pooling2d_35/MaxPool:output:04sequential_7/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_7/conv2d_36/Conv2Dм
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpт
sequential_7/conv2d_36/BiasAddBiasAdd&sequential_7/conv2d_36/Conv2D:output:05sequential_7/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_7/conv2d_36/BiasAddд
sequential_7/conv2d_36/ReluRelu'sequential_7/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_7/conv2d_36/ReluЫ
%sequential_7/max_pooling2d_36/MaxPoolMaxPool)sequential_7/conv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_36/MaxPool▄
,sequential_7/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_37/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_37/Conv2DConv2D.sequential_7/max_pooling2d_36/MaxPool:output:04sequential_7/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_37/Conv2Dм
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpт
sequential_7/conv2d_37/BiasAddBiasAdd&sequential_7/conv2d_37/Conv2D:output:05sequential_7/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_37/BiasAddд
sequential_7/conv2d_37/ReluRelu'sequential_7/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_37/ReluЫ
%sequential_7/max_pooling2d_37/MaxPoolMaxPool)sequential_7/conv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_37/MaxPool▄
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_38/Conv2DConv2D.sequential_7/max_pooling2d_37/MaxPool:output:04sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2Dм
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpт
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_7/conv2d_38/BiasAddд
sequential_7/conv2d_38/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_7/conv2d_38/ReluЫ
%sequential_7/max_pooling2d_38/MaxPoolMaxPool)sequential_7/conv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPool▄
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2Dм
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpт
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_39/BiasAddд
sequential_7/conv2d_39/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_39/ReluЫ
%sequential_7/max_pooling2d_39/MaxPoolMaxPool)sequential_7/conv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPoolЊ
%sequential_7/dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2'
%sequential_7/dropout_21/dropout/ConstВ
#sequential_7/dropout_21/dropout/MulMul.sequential_7/max_pooling2d_39/MaxPool:output:0.sequential_7/dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2%
#sequential_7/dropout_21/dropout/Mulг
%sequential_7/dropout_21/dropout/ShapeShape.sequential_7/max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_21/dropout/ShapeЁ
<sequential_7/dropout_21/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02>
<sequential_7/dropout_21/dropout/random_uniform/RandomUniformЦ
.sequential_7/dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_7/dropout_21/dropout/GreaterEqual/yД
,sequential_7/dropout_21/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_21/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2.
,sequential_7/dropout_21/dropout/GreaterEqualл
$sequential_7/dropout_21/dropout/CastCast0sequential_7/dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2&
$sequential_7/dropout_21/dropout/Castс
%sequential_7/dropout_21/dropout/Mul_1Mul'sequential_7/dropout_21/dropout/Mul:z:0(sequential_7/dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2'
%sequential_7/dropout_21/dropout/Mul_1Ї
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Constл
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/dropout/Mul_1:z:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2 
sequential_7/flatten_7/ReshapeЛ
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOpО
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/MatMul¤
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/BiasAddЏ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/ReluЊ
%sequential_7/dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_22/dropout/Constя
#sequential_7/dropout_22/dropout/MulMul(sequential_7/dense_21/Relu:activations:0.sequential_7/dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_7/dropout_22/dropout/Mulд
%sequential_7/dropout_22/dropout/ShapeShape(sequential_7/dense_21/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_22/dropout/Shape§
<sequential_7/dropout_22/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_7/dropout_22/dropout/random_uniform/RandomUniformЦ
.sequential_7/dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_22/dropout/GreaterEqual/yЪ
,sequential_7/dropout_22/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_22/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_7/dropout_22/dropout/GreaterEqual╚
$sequential_7/dropout_22/dropout/CastCast0sequential_7/dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_7/dropout_22/dropout/Cast█
%sequential_7/dropout_22/dropout/Mul_1Mul'sequential_7/dropout_22/dropout/Mul:z:0(sequential_7/dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_7/dropout_22/dropout/Mul_1Л
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/dropout/Mul_1:z:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/MatMul¤
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/BiasAddЏ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/ReluЊ
%sequential_7/dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_23/dropout/Constя
#sequential_7/dropout_23/dropout/MulMul(sequential_7/dense_22/Relu:activations:0.sequential_7/dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_7/dropout_23/dropout/Mulд
%sequential_7/dropout_23/dropout/ShapeShape(sequential_7/dense_22/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_23/dropout/Shape§
<sequential_7/dropout_23/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_7/dropout_23/dropout/random_uniform/RandomUniformЦ
.sequential_7/dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_23/dropout/GreaterEqual/yЪ
,sequential_7/dropout_23/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_23/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_7/dropout_23/dropout/GreaterEqual╚
$sequential_7/dropout_23/dropout/CastCast0sequential_7/dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_7/dropout_23/dropout/Cast█
%sequential_7/dropout_23/dropout/Mul_1Mul'sequential_7/dropout_23/dropout/Mul:z:0(sequential_7/dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_7/dropout_23/dropout/Mul_1Е
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЦ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/SoftmaxТ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulП
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulП
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul├

IdentityIdentitydense_23/Softmax:softmax:03^conv2d_35/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp2^sequential_7/batch_normalization_7/AssignNewValue4^sequential_7/batch_normalization_7/AssignNewValue_1C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_35/BiasAdd/ReadVariableOp-^sequential_7/conv2d_35/Conv2D/ReadVariableOp.^sequential_7/conv2d_36/BiasAdd/ReadVariableOp-^sequential_7/conv2d_36/Conv2D/ReadVariableOp.^sequential_7/conv2d_37/BiasAdd/ReadVariableOp-^sequential_7/conv2d_37/Conv2D/ReadVariableOp.^sequential_7/conv2d_38/BiasAdd/ReadVariableOp-^sequential_7/conv2d_38/Conv2D/ReadVariableOp.^sequential_7/conv2d_39/BiasAdd/ReadVariableOp-^sequential_7/conv2d_39/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2f
1sequential_7/batch_normalization_7/AssignNewValue1sequential_7/batch_normalization_7/AssignNewValue2j
3sequential_7/batch_normalization_7/AssignNewValue_13sequential_7/batch_normalization_7/AssignNewValue_12ѕ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp-sequential_7/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_35/Conv2D/ReadVariableOp,sequential_7/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_36/BiasAdd/ReadVariableOp-sequential_7/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_36/Conv2D/ReadVariableOp,sequential_7/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_37/BiasAdd/ReadVariableOp-sequential_7/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_37/Conv2D/ReadVariableOp,sequential_7/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_38/BiasAdd/ReadVariableOp-sequential_7/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_38/Conv2D/ReadVariableOp,sequential_7/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_39/BiasAdd/ReadVariableOp-sequential_7/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_39/Conv2D/ReadVariableOp,sequential_7/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╣
г
D__inference_dense_21_layer_call_and_return_conditional_losses_934992

inputs2
matmul_readvariableop_resource:
ђ ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
ReluК
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ 
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_36_layer_call_fn_934784

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_9347782
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
 3
н
@__inference_CNN3_layer_call_and_return_conditional_losses_935896

inputs!
sequential_7_935835:!
sequential_7_935837:!
sequential_7_935839:!
sequential_7_935841:-
sequential_7_935843: !
sequential_7_935845: .
sequential_7_935847: ђ"
sequential_7_935849:	ђ/
sequential_7_935851:ђђ"
sequential_7_935853:	ђ/
sequential_7_935855:ђђ"
sequential_7_935857:	ђ/
sequential_7_935859:ђђ"
sequential_7_935861:	ђ'
sequential_7_935863:
ђ ђ"
sequential_7_935865:	ђ'
sequential_7_935867:
ђђ"
sequential_7_935869:	ђ"
dense_23_935872:	ђ
dense_23_935874:
identityѕб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpб dense_23/StatefulPartitionedCallб$sequential_7/StatefulPartitionedCallю
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7_935835sequential_7_935837sequential_7_935839sequential_7_935841sequential_7_935843sequential_7_935845sequential_7_935847sequential_7_935849sequential_7_935851sequential_7_935853sequential_7_935855sequential_7_935857sequential_7_935859sequential_7_935861sequential_7_935863sequential_7_935865sequential_7_935867sequential_7_935869*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_9354202&
$sequential_7/StatefulPartitionedCall└
 dense_23/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0dense_23_935872dense_23_935874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_9357072"
 dense_23/StatefulPartitionedCall─
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_935843*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mul╝
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_935863* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mul╝
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_935867* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mulС
IdentityIdentity)dense_23/StatefulPartitionedCall:output:03^conv2d_35/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp!^dense_23/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ж
G
+__inference_dropout_21_layer_call_fn_937736

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_9349652
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Л
б
*__inference_conv2d_39_layer_call_fn_937720

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_39_layer_call_and_return_conditional_losses_9349532
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Н
d
+__inference_dropout_23_layer_call_fn_937870

inputs
identityѕбStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_9351132
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_38_layer_call_fn_934808

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_9348022
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
кЇ
Д
H__inference_sequential_7_layer_call_and_return_conditional_losses_937110

inputs;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_35_conv2d_readvariableop_resource: 7
)conv2d_35_biasadd_readvariableop_resource: C
(conv2d_36_conv2d_readvariableop_resource: ђ8
)conv2d_36_biasadd_readvariableop_resource:	ђD
(conv2d_37_conv2d_readvariableop_resource:ђђ8
)conv2d_37_biasadd_readvariableop_resource:	ђD
(conv2d_38_conv2d_readvariableop_resource:ђђ8
)conv2d_38_biasadd_readvariableop_resource:	ђD
(conv2d_39_conv2d_readvariableop_resource:ђђ8
)conv2d_39_biasadd_readvariableop_resource:	ђ;
'dense_21_matmul_readvariableop_resource:
ђ ђ7
(dense_21_biasadd_readvariableop_resource:	ђ;
'dense_22_matmul_readvariableop_resource:
ђђ7
(dense_22_biasadd_readvariableop_resource:	ђ
identityѕб5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б conv2d_35/BiasAdd/ReadVariableOpбconv2d_35/Conv2D/ReadVariableOpб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб conv2d_36/BiasAdd/ReadVariableOpбconv2d_36/Conv2D/ReadVariableOpб conv2d_37/BiasAdd/ReadVariableOpбconv2d_37/Conv2D/ReadVariableOpб conv2d_38/BiasAdd/ReadVariableOpбconv2d_38/Conv2D/ReadVariableOpб conv2d_39/BiasAdd/ReadVariableOpбconv2d_39/Conv2D/ReadVariableOpбdense_21/BiasAdd/ReadVariableOpбdense_21/MatMul/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЎ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Ў
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2ф
lambda_7/strided_sliceStridedSliceinputs%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_7/strided_sliceХ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1ж
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1у
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3│
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_35/Conv2D/ReadVariableOpт
conv2d_35/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_35/Conv2Dф
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp░
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_35/BiasAdd~
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_35/Relu╩
max_pooling2d_35/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_35/MaxPool┤
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_36/Conv2D/ReadVariableOpП
conv2d_36/Conv2DConv2D!max_pooling2d_35/MaxPool:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_36/Conv2DФ
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp▒
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_36/BiasAdd
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_36/Relu╦
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPoolх
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_37/Conv2D/ReadVariableOpП
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_37/Conv2DФ
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp▒
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_37/Relu╦
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPoolх
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_38/Conv2D/ReadVariableOpП
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
conv2d_38/Conv2DФ
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
conv2d_38/Relu╦
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolх
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_39/Conv2D/ReadVariableOpП
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_39/Conv2DФ
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▒
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_39/Relu╦
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPoolћ
dropout_21/IdentityIdentity!max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2
dropout_21/Identitys
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_7/Constю
flatten_7/ReshapeReshapedropout_21/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2
flatten_7/Reshapeф
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02 
dense_21/MatMul/ReadVariableOpБ
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_21/MatMulе
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_21/BiasAdd/ReadVariableOpд
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_21/BiasAddt
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_21/Reluє
dropout_22/IdentityIdentitydense_21/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_22/Identityф
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_22/MatMul/ReadVariableOpЦ
dense_22/MatMulMatMuldropout_22/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_22/MatMulе
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_22/BiasAdd/ReadVariableOpд
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_22/BiasAddt
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_22/Reluє
dropout_23/IdentityIdentitydense_22/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_23/Identity┘
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulл
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulл
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul»
IdentityIdentitydropout_23/Identity:output:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
э
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_935271

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
х
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_935146

inputs
identityѕc
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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┬
`
D__inference_lambda_7_layer_call_and_return_conditional_losses_937487

inputs
identityЃ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЄ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1Є
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2§
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╔▓
ф
@__inference_CNN3_layer_call_and_return_conditional_losses_936469

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_35_conv2d_readvariableop_resource: D
6sequential_7_conv2d_35_biasadd_readvariableop_resource: P
5sequential_7_conv2d_36_conv2d_readvariableop_resource: ђE
6sequential_7_conv2d_36_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_37_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_37_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_38_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_38_biasadd_readvariableop_resource:	ђQ
5sequential_7_conv2d_39_conv2d_readvariableop_resource:ђђE
6sequential_7_conv2d_39_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_21_matmul_readvariableop_resource:
ђ ђD
5sequential_7_dense_21_biasadd_readvariableop_resource:	ђH
4sequential_7_dense_22_matmul_readvariableop_resource:
ђђD
5sequential_7_dense_22_biasadd_readvariableop_resource:	ђ:
'dense_23_matmul_readvariableop_resource:	ђ6
(dense_23_biasadd_readvariableop_resource:
identityѕб2conv2d_35/kernel/Regularizer/Square/ReadVariableOpб1dense_21/kernel/Regularizer/Square/ReadVariableOpб1dense_22/kernel/Regularizer/Square/ReadVariableOpбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpбBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_7/batch_normalization_7/ReadVariableOpб3sequential_7/batch_normalization_7/ReadVariableOp_1б-sequential_7/conv2d_35/BiasAdd/ReadVariableOpб,sequential_7/conv2d_35/Conv2D/ReadVariableOpб-sequential_7/conv2d_36/BiasAdd/ReadVariableOpб,sequential_7/conv2d_36/Conv2D/ReadVariableOpб-sequential_7/conv2d_37/BiasAdd/ReadVariableOpб,sequential_7/conv2d_37/Conv2D/ReadVariableOpб-sequential_7/conv2d_38/BiasAdd/ReadVariableOpб,sequential_7/conv2d_38/Conv2D/ReadVariableOpб-sequential_7/conv2d_39/BiasAdd/ReadVariableOpб,sequential_7/conv2d_39/Conv2D/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOp»
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2в
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_sliceП
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpс
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1љ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpќ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_35/Conv2D/ReadVariableOpЎ
sequential_7/conv2d_35/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_35/Conv2DЛ
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_35/BiasAdd/ReadVariableOpС
sequential_7/conv2d_35/BiasAddBiasAdd&sequential_7/conv2d_35/Conv2D:output:05sequential_7/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_35/BiasAddЦ
sequential_7/conv2d_35/ReluRelu'sequential_7/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_35/Reluы
%sequential_7/max_pooling2d_35/MaxPoolMaxPool)sequential_7/conv2d_35/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_35/MaxPool█
,sequential_7/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_7/conv2d_36/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_36/Conv2DConv2D.sequential_7/max_pooling2d_35/MaxPool:output:04sequential_7/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_7/conv2d_36/Conv2Dм
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_36/BiasAdd/ReadVariableOpт
sequential_7/conv2d_36/BiasAddBiasAdd&sequential_7/conv2d_36/Conv2D:output:05sequential_7/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_7/conv2d_36/BiasAddд
sequential_7/conv2d_36/ReluRelu'sequential_7/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_7/conv2d_36/ReluЫ
%sequential_7/max_pooling2d_36/MaxPoolMaxPool)sequential_7/conv2d_36/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_36/MaxPool▄
,sequential_7/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_37/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_37/Conv2DConv2D.sequential_7/max_pooling2d_36/MaxPool:output:04sequential_7/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_37/Conv2Dм
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_37/BiasAdd/ReadVariableOpт
sequential_7/conv2d_37/BiasAddBiasAdd&sequential_7/conv2d_37/Conv2D:output:05sequential_7/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_37/BiasAddд
sequential_7/conv2d_37/ReluRelu'sequential_7/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_37/ReluЫ
%sequential_7/max_pooling2d_37/MaxPoolMaxPool)sequential_7/conv2d_37/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_37/MaxPool▄
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_38/Conv2DConv2D.sequential_7/max_pooling2d_37/MaxPool:output:04sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2Dм
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpт
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_7/conv2d_38/BiasAddд
sequential_7/conv2d_38/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_7/conv2d_38/ReluЫ
%sequential_7/max_pooling2d_38/MaxPoolMaxPool)sequential_7/conv2d_38/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPool▄
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpЉ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2Dм
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpт
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_7/conv2d_39/BiasAddд
sequential_7/conv2d_39/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_7/conv2d_39/ReluЫ
%sequential_7/max_pooling2d_39/MaxPoolMaxPool)sequential_7/conv2d_39/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPool╗
 sequential_7/dropout_21/IdentityIdentity.sequential_7/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2"
 sequential_7/dropout_21/IdentityЇ
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_7/flatten_7/Constл
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_21/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         ђ 2 
sequential_7/flatten_7/ReshapeЛ
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype02-
+sequential_7/dense_21/MatMul/ReadVariableOpО
sequential_7/dense_21/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/MatMul¤
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_21/BiasAdd/ReadVariableOp┌
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/BiasAddЏ
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_21/ReluГ
 sequential_7/dropout_22/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_7/dropout_22/IdentityЛ
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_7/dense_22/MatMul/ReadVariableOp┘
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_22/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/MatMul¤
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_7/dense_22/BiasAdd/ReadVariableOp┌
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/BiasAddЏ
sequential_7/dense_22/ReluRelu&sequential_7/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_7/dense_22/ReluГ
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_22/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_7/dropout_23/IdentityЕ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_23/MatMul/ReadVariableOp▒
dense_23/MatMulMatMul)sequential_7/dropout_23/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpЦ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/SoftmaxТ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_35/kernel/Regularizer/SquareА
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const┬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumЇ
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_35/kernel/Regularizer/mul/x─
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulП
1dense_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ђ ђ*
dtype023
1dense_21/kernel/Regularizer/Square/ReadVariableOpИ
"dense_21/kernel/Regularizer/SquareSquare9dense_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђ ђ2$
"dense_21/kernel/Regularizer/SquareЌ
!dense_21/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_21/kernel/Regularizer/ConstЙ
dense_21/kernel/Regularizer/SumSum&dense_21/kernel/Regularizer/Square:y:0*dense_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/SumІ
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_21/kernel/Regularizer/mul/x└
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0(dense_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_21/kernel/Regularizer/mulП
1dense_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_22/kernel/Regularizer/Square/ReadVariableOpИ
"dense_22/kernel/Regularizer/SquareSquare9dense_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_22/kernel/Regularizer/SquareЌ
!dense_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_22/kernel/Regularizer/ConstЙ
dense_22/kernel/Regularizer/SumSum&dense_22/kernel/Regularizer/Square:y:0*dense_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/SumІ
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_22/kernel/Regularizer/mul/x└
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0(dense_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_22/kernel/Regularizer/mul┘	
IdentityIdentitydense_23/Softmax:softmax:03^conv2d_35/kernel/Regularizer/Square/ReadVariableOp2^dense_21/kernel/Regularizer/Square/ReadVariableOp2^dense_22/kernel/Regularizer/Square/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_35/BiasAdd/ReadVariableOp-^sequential_7/conv2d_35/Conv2D/ReadVariableOp.^sequential_7/conv2d_36/BiasAdd/ReadVariableOp-^sequential_7/conv2d_36/Conv2D/ReadVariableOp.^sequential_7/conv2d_37/BiasAdd/ReadVariableOp-^sequential_7/conv2d_37/Conv2D/ReadVariableOp.^sequential_7/conv2d_38/BiasAdd/ReadVariableOp-^sequential_7/conv2d_38/Conv2D/ReadVariableOp.^sequential_7/conv2d_39/BiasAdd/ReadVariableOp-^sequential_7/conv2d_39/Conv2D/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2f
1dense_21/kernel/Regularizer/Square/ReadVariableOp1dense_21/kernel/Regularizer/Square/ReadVariableOp2f
1dense_22/kernel/Regularizer/Square/ReadVariableOp1dense_22/kernel/Regularizer/Square/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2ѕ
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_35/BiasAdd/ReadVariableOp-sequential_7/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_35/Conv2D/ReadVariableOp,sequential_7/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_36/BiasAdd/ReadVariableOp-sequential_7/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_36/Conv2D/ReadVariableOp,sequential_7/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_37/BiasAdd/ReadVariableOp-sequential_7/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_37/Conv2D/ReadVariableOp,sequential_7/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_38/BiasAdd/ReadVariableOp-sequential_7/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_38/Conv2D/ReadVariableOp,sequential_7/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_39/BiasAdd/ReadVariableOp-sequential_7/conv2d_39/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_39/Conv2D/ReadVariableOp,sequential_7/conv2d_39/Conv2D/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_37_layer_call_fn_934796

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_9347902
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
А
Ђ
E__inference_conv2d_39_layer_call_and_return_conditional_losses_937731

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
р
┤
%__inference_CNN3_layer_call_fn_936363
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђ ђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:
identityѕбStatefulPartitionedCallу
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
:         *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_CNN3_layer_call_and_return_conditional_losses_9358962
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЪ
C
input_18
serving_default_input_1:0         KK<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:░Ч
Ђ

h2ptjl
_output
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Ю__call__
+ъ&call_and_return_all_conditional_losses
Ъ_default_save_signature
	аcall"Ј	
_tf_keras_modelш{"name": "CNN3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN3", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN3"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
пѕ
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
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
regularization_losses
trainable_variables
	variables
	keras_api
А__call__
+б&call_and_return_all_conditional_losses"ёё
_tf_keras_sequentialСЃ{"name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_7_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 42, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_7_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_7_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 29}, {"class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 30}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 31}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 34}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 41}]}}}
О

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
Б__call__
+ц&call_and_return_all_conditional_losses"░
_tf_keras_layerќ{"name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
╗
%iter

&beta_1

'beta_2
	(decay
)learning_ratemщ mЩ*mч+mЧ,m§-m■.m /mђ0mЂ1mѓ2mЃ3mё4mЁ5mє6mЄ7mѕ8mЅ9mіvІ vї*vЇ+vј,vЈ-vљ.vЉ/vњ0vЊ1vћ2vЋ3vќ4vЌ5vў6vЎ7vџ8vЏ9vю"
	optimizer
 "
trackable_list_wrapper
д
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
16
 17"
trackable_list_wrapper
Х
*0
+1
:2
;3
,4
-5
.6
/7
08
19
210
311
412
513
614
715
816
917
18
 19"
trackable_list_wrapper
╬
<metrics
=layer_metrics

>layers
?layer_regularization_losses
regularization_losses
trainable_variables
	variables
@non_trainable_variables
Ю__call__
Ъ_default_save_signature
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
-
Цserving_default"
signature_map
п
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
д__call__
+Д&call_and_return_all_conditional_losses"К
_tf_keras_layerГ{"name": "lambda_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
─

Eaxis
	*gamma
+beta
:moving_mean
;moving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
е__call__
+Е&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
б

,kernel
-bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses"ч	
_tf_keras_layerр	{"name": "conv2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
│
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
г__call__
+Г&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"name": "max_pooling2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 49}}
о


.kernel
/bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
«__call__
+»&call_and_return_all_conditional_losses"»	
_tf_keras_layerЋ	{"name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
│
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"name": "max_pooling2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 51}}
п


0kernel
1bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"▒	
_tf_keras_layerЌ	{"name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
│
^regularization_losses
_trainable_variables
`	variables
a	keras_api
┤__call__
+х&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"name": "max_pooling2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 53}}
о


2kernel
3bias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
Х__call__
+и&call_and_return_all_conditional_losses"»	
_tf_keras_layerЋ	{"name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 9, 9, 256]}}
│
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
И__call__
+╣&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"name": "max_pooling2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 55}}
О


4kernel
5bias
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
║__call__
+╗&call_and_return_all_conditional_losses"░	
_tf_keras_layerќ	{"name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 4, 4, 512]}}
│
nregularization_losses
otrainable_variables
p	variables
q	keras_api
╝__call__
+й&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"name": "max_pooling2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 57}}
Ђ
rregularization_losses
strainable_variables
t	variables
u	keras_api
Й__call__
+┐&call_and_return_all_conditional_losses"­
_tf_keras_layerо{"name": "dropout_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 30}
ў
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"Є
_tf_keras_layerь{"name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 58}}
е	

6kernel
7bias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
┬__call__
+├&call_and_return_all_conditional_losses"Ђ
_tf_keras_layerу{"name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 34}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 4096]}}
Ѓ
~regularization_losses
trainable_variables
ђ	variables
Ђ	keras_api
─__call__
+┼&call_and_return_all_conditional_losses"­
_tf_keras_layerо{"name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 36}
ф	

8kernel
9bias
ѓregularization_losses
Ѓtrainable_variables
ё	variables
Ё	keras_api
к__call__
+К&call_and_return_all_conditional_losses" 
_tf_keras_layerт{"name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Ё
єregularization_losses
Єtrainable_variables
ѕ	variables
Ѕ	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses"­
_tf_keras_layerо{"name": "dropout_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 41}
8
╩0
╦1
╠2"
trackable_list_wrapper
ќ
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
915"
trackable_list_wrapper
д
*0
+1
:2
;3
,4
-5
.6
/7
08
19
210
311
412
513
614
715
816
917"
trackable_list_wrapper
х
іmetrics
Іlayer_metrics
їlayers
 Їlayer_regularization_losses
regularization_losses
trainable_variables
	variables
јnon_trainable_variables
А__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
": 	ђ2dense_23/kernel
:2dense_23/bias
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
х
Јmetrics
љlayers
Љlayer_metrics
 њlayer_regularization_losses
!regularization_losses
"trainable_variables
#	variables
Њnon_trainable_variables
Б__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
*:( 2conv2d_35/kernel
: 2conv2d_35/bias
+:) ђ2conv2d_36/kernel
:ђ2conv2d_36/bias
,:*ђђ2conv2d_37/kernel
:ђ2conv2d_37/bias
,:*ђђ2conv2d_38/kernel
:ђ2conv2d_38/bias
,:*ђђ2conv2d_39/kernel
:ђ2conv2d_39/bias
#:!
ђ ђ2dense_21/kernel
:ђ2dense_21/bias
#:!
ђђ2dense_22/kernel
:ђ2dense_22/bias
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
0
ћ0
Ћ1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ќmetrics
Ќlayers
ўlayer_metrics
 Ўlayer_regularization_losses
Aregularization_losses
Btrainable_variables
C	variables
џnon_trainable_variables
д__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
<
*0
+1
:2
;3"
trackable_list_wrapper
х
Џmetrics
юlayers
Юlayer_metrics
 ъlayer_regularization_losses
Fregularization_losses
Gtrainable_variables
H	variables
Ъnon_trainable_variables
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
(
╩0"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
х
аmetrics
Аlayers
бlayer_metrics
 Бlayer_regularization_losses
Jregularization_losses
Ktrainable_variables
L	variables
цnon_trainable_variables
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Цmetrics
дlayers
Дlayer_metrics
 еlayer_regularization_losses
Nregularization_losses
Otrainable_variables
P	variables
Еnon_trainable_variables
г__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
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
х
фmetrics
Фlayers
гlayer_metrics
 Гlayer_regularization_losses
Rregularization_losses
Strainable_variables
T	variables
«non_trainable_variables
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
»metrics
░layers
▒layer_metrics
 ▓layer_regularization_losses
Vregularization_losses
Wtrainable_variables
X	variables
│non_trainable_variables
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
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
х
┤metrics
хlayers
Хlayer_metrics
 иlayer_regularization_losses
Zregularization_losses
[trainable_variables
\	variables
Иnon_trainable_variables
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
╣metrics
║layers
╗layer_metrics
 ╝layer_regularization_losses
^regularization_losses
_trainable_variables
`	variables
йnon_trainable_variables
┤__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
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
х
Йmetrics
┐layers
└layer_metrics
 ┴layer_regularization_losses
bregularization_losses
ctrainable_variables
d	variables
┬non_trainable_variables
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
├metrics
─layers
┼layer_metrics
 кlayer_regularization_losses
fregularization_losses
gtrainable_variables
h	variables
Кnon_trainable_variables
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
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
х
╚metrics
╔layers
╩layer_metrics
 ╦layer_regularization_losses
jregularization_losses
ktrainable_variables
l	variables
╠non_trainable_variables
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
═metrics
╬layers
¤layer_metrics
 лlayer_regularization_losses
nregularization_losses
otrainable_variables
p	variables
Лnon_trainable_variables
╝__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
мmetrics
Мlayers
нlayer_metrics
 Нlayer_regularization_losses
rregularization_losses
strainable_variables
t	variables
оnon_trainable_variables
Й__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Оmetrics
пlayers
┘layer_metrics
 ┌layer_regularization_losses
vregularization_losses
wtrainable_variables
x	variables
█non_trainable_variables
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
(
╦0"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
х
▄metrics
Пlayers
яlayer_metrics
 ▀layer_regularization_losses
zregularization_losses
{trainable_variables
|	variables
Яnon_trainable_variables
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Х
рmetrics
Рlayers
сlayer_metrics
 Сlayer_regularization_losses
~regularization_losses
trainable_variables
ђ	variables
тnon_trainable_variables
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
(
╠0"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
И
Тmetrics
уlayers
Уlayer_metrics
 жlayer_regularization_losses
ѓregularization_losses
Ѓtrainable_variables
ё	variables
Жnon_trainable_variables
к__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
вmetrics
Вlayers
ьlayer_metrics
 Ьlayer_regularization_losses
єregularization_losses
Єtrainable_variables
ѕ	variables
№non_trainable_variables
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
д
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
:0
;1"
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
п

­total

ыcount
Ы	variables
з	keras_api"Ю
_tf_keras_metricѓ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 61}
Џ

Зtotal

шcount
Ш
_fn_kwargs
э	variables
Э	keras_api"¤
_tf_keras_metric┤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
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
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
╩0"
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
╦0"
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
╠0"
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
­0
ы1"
trackable_list_wrapper
.
Ы	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
З0
ш1"
trackable_list_wrapper
.
э	variables"
_generic_user_object
':%	ђ2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
.:,2"Adam/batch_normalization_7/gamma/m
-:+2!Adam/batch_normalization_7/beta/m
/:- 2Adam/conv2d_35/kernel/m
!: 2Adam/conv2d_35/bias/m
0:. ђ2Adam/conv2d_36/kernel/m
": ђ2Adam/conv2d_36/bias/m
1:/ђђ2Adam/conv2d_37/kernel/m
": ђ2Adam/conv2d_37/bias/m
1:/ђђ2Adam/conv2d_38/kernel/m
": ђ2Adam/conv2d_38/bias/m
1:/ђђ2Adam/conv2d_39/kernel/m
": ђ2Adam/conv2d_39/bias/m
(:&
ђ ђ2Adam/dense_21/kernel/m
!:ђ2Adam/dense_21/bias/m
(:&
ђђ2Adam/dense_22/kernel/m
!:ђ2Adam/dense_22/bias/m
':%	ђ2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v
.:,2"Adam/batch_normalization_7/gamma/v
-:+2!Adam/batch_normalization_7/beta/v
/:- 2Adam/conv2d_35/kernel/v
!: 2Adam/conv2d_35/bias/v
0:. ђ2Adam/conv2d_36/kernel/v
": ђ2Adam/conv2d_36/bias/v
1:/ђђ2Adam/conv2d_37/kernel/v
": ђ2Adam/conv2d_37/bias/v
1:/ђђ2Adam/conv2d_38/kernel/v
": ђ2Adam/conv2d_38/bias/v
1:/ђђ2Adam/conv2d_39/kernel/v
": ђ2Adam/conv2d_39/bias/v
(:&
ђ ђ2Adam/dense_21/kernel/v
!:ђ2Adam/dense_21/bias/v
(:&
ђђ2Adam/dense_22/kernel/v
!:ђ2Adam/dense_22/bias/v
о2М
%__inference_CNN3_layer_call_fn_936228
%__inference_CNN3_layer_call_fn_936273
%__inference_CNN3_layer_call_fn_936318
%__inference_CNN3_layer_call_fn_936363┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┬2┐
@__inference_CNN3_layer_call_and_return_conditional_losses_936469
@__inference_CNN3_layer_call_and_return_conditional_losses_936596
@__inference_CNN3_layer_call_and_return_conditional_losses_936702
@__inference_CNN3_layer_call_and_return_conditional_losses_936829┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
у2С
!__inference__wrapped_model_934634Й
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *.б+
)і&
input_1         KK
ё2Ђ
__inference_call_889595
__inference_call_889683
__inference_call_889771│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓ2 
-__inference_sequential_7_layer_call_fn_936888
-__inference_sequential_7_layer_call_fn_936929
-__inference_sequential_7_layer_call_fn_936970
-__inference_sequential_7_layer_call_fn_937011└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
H__inference_sequential_7_layer_call_and_return_conditional_losses_937110
H__inference_sequential_7_layer_call_and_return_conditional_losses_937230
H__inference_sequential_7_layer_call_and_return_conditional_losses_937329
H__inference_sequential_7_layer_call_and_return_conditional_losses_937449└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_23_layer_call_fn_937458б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_23_layer_call_and_return_conditional_losses_937469б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╦B╚
$__inference_signature_wrapper_936183input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю2Ў
)__inference_lambda_7_layer_call_fn_937474
)__inference_lambda_7_layer_call_fn_937479└
и▓│
FullArgSpec1
args)џ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsџ

 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
D__inference_lambda_7_layer_call_and_return_conditional_losses_937487
D__inference_lambda_7_layer_call_and_return_conditional_losses_937495└
и▓│
FullArgSpec1
args)џ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsџ

 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_7_layer_call_fn_937508
6__inference_batch_normalization_7_layer_call_fn_937521
6__inference_batch_normalization_7_layer_call_fn_937534
6__inference_batch_normalization_7_layer_call_fn_937547┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937565
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937583
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937601
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937619┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
н2Л
*__inference_conv2d_35_layer_call_fn_937634б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_35_layer_call_and_return_conditional_losses_937651б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_35_layer_call_fn_934772Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_934766Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
*__inference_conv2d_36_layer_call_fn_937660б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_36_layer_call_and_return_conditional_losses_937671б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_36_layer_call_fn_934784Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_934778Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
*__inference_conv2d_37_layer_call_fn_937680б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_37_layer_call_and_return_conditional_losses_937691б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_37_layer_call_fn_934796Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_934790Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
*__inference_conv2d_38_layer_call_fn_937700б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_38_layer_call_and_return_conditional_losses_937711б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_38_layer_call_fn_934808Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_934802Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
*__inference_conv2d_39_layer_call_fn_937720б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_39_layer_call_and_return_conditional_losses_937731б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_39_layer_call_fn_934820Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_934814Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ћ2Љ
+__inference_dropout_21_layer_call_fn_937736
+__inference_dropout_21_layer_call_fn_937741┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_21_layer_call_and_return_conditional_losses_937746
F__inference_dropout_21_layer_call_and_return_conditional_losses_937758┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
н2Л
*__inference_flatten_7_layer_call_fn_937763б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_flatten_7_layer_call_and_return_conditional_losses_937769б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_21_layer_call_fn_937784б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_21_layer_call_and_return_conditional_losses_937801б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_22_layer_call_fn_937806
+__inference_dropout_22_layer_call_fn_937811┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_22_layer_call_and_return_conditional_losses_937816
F__inference_dropout_22_layer_call_and_return_conditional_losses_937828┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_22_layer_call_fn_937843б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_22_layer_call_and_return_conditional_losses_937860б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_23_layer_call_fn_937865
+__inference_dropout_23_layer_call_fn_937870┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_23_layer_call_and_return_conditional_losses_937875
F__inference_dropout_23_layer_call_and_return_conditional_losses_937887┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
│2░
__inference_loss_fn_0_937898Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
│2░
__inference_loss_fn_1_937909Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
│2░
__inference_loss_fn_2_937920Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б Й
@__inference_CNN3_layer_call_and_return_conditional_losses_936469z*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p 
ф "%б"
і
0         
џ Й
@__inference_CNN3_layer_call_and_return_conditional_losses_936596z*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p
ф "%б"
і
0         
џ ┐
@__inference_CNN3_layer_call_and_return_conditional_losses_936702{*+:;,-./0123456789 <б9
2б/
)і&
input_1         KK
p 
ф "%б"
і
0         
џ ┐
@__inference_CNN3_layer_call_and_return_conditional_losses_936829{*+:;,-./0123456789 <б9
2б/
)і&
input_1         KK
p
ф "%б"
і
0         
џ Ќ
%__inference_CNN3_layer_call_fn_936228n*+:;,-./0123456789 <б9
2б/
)і&
input_1         KK
p 
ф "і         ќ
%__inference_CNN3_layer_call_fn_936273m*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p 
ф "і         ќ
%__inference_CNN3_layer_call_fn_936318m*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p
ф "і         Ќ
%__inference_CNN3_layer_call_fn_936363n*+:;,-./0123456789 <б9
2б/
)і&
input_1         KK
p
ф "і         Ф
!__inference__wrapped_model_934634Ё*+:;,-./0123456789 8б5
.б+
)і&
input_1         KK
ф "3ф0
.
output_1"і
output_1         В
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937565ќ*+:;MбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ В
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937583ќ*+:;MбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ К
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937601r*+:;;б8
1б.
(і%
inputs         KK
p 
ф "-б*
#і 
0         KK
џ К
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937619r*+:;;б8
1б.
(і%
inputs         KK
p
ф "-б*
#і 
0         KK
џ ─
6__inference_batch_normalization_7_layer_call_fn_937508Ѕ*+:;MбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ─
6__inference_batch_normalization_7_layer_call_fn_937521Ѕ*+:;MбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           Ъ
6__inference_batch_normalization_7_layer_call_fn_937534e*+:;;б8
1б.
(і%
inputs         KK
p 
ф " і         KKЪ
6__inference_batch_normalization_7_layer_call_fn_937547e*+:;;б8
1б.
(і%
inputs         KK
p
ф " і         KKx
__inference_call_889595]*+:;,-./0123456789 3б0
)б&
 і
inputsђKK
p
ф "і	ђx
__inference_call_889683]*+:;,-./0123456789 3б0
)б&
 і
inputsђKK
p 
ф "і	ђѕ
__inference_call_889771m*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p 
ф "і         х
E__inference_conv2d_35_layer_call_and_return_conditional_losses_937651l,-7б4
-б*
(і%
inputs         KK
ф "-б*
#і 
0         KK 
џ Ї
*__inference_conv2d_35_layer_call_fn_937634_,-7б4
-б*
(і%
inputs         KK
ф " і         KK Х
E__inference_conv2d_36_layer_call_and_return_conditional_losses_937671m./7б4
-б*
(і%
inputs         %% 
ф ".б+
$і!
0         %%ђ
џ ј
*__inference_conv2d_36_layer_call_fn_937660`./7б4
-б*
(і%
inputs         %% 
ф "!і         %%ђи
E__inference_conv2d_37_layer_call_and_return_conditional_losses_937691n018б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ Ј
*__inference_conv2d_37_layer_call_fn_937680a018б5
.б+
)і&
inputs         ђ
ф "!і         ђи
E__inference_conv2d_38_layer_call_and_return_conditional_losses_937711n238б5
.б+
)і&
inputs         		ђ
ф ".б+
$і!
0         		ђ
џ Ј
*__inference_conv2d_38_layer_call_fn_937700a238б5
.б+
)і&
inputs         		ђ
ф "!і         		ђи
E__inference_conv2d_39_layer_call_and_return_conditional_losses_937731n458б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ Ј
*__inference_conv2d_39_layer_call_fn_937720a458б5
.б+
)і&
inputs         ђ
ф "!і         ђд
D__inference_dense_21_layer_call_and_return_conditional_losses_937801^670б-
&б#
!і
inputs         ђ 
ф "&б#
і
0         ђ
џ ~
)__inference_dense_21_layer_call_fn_937784Q670б-
&б#
!і
inputs         ђ 
ф "і         ђд
D__inference_dense_22_layer_call_and_return_conditional_losses_937860^890б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ~
)__inference_dense_22_layer_call_fn_937843Q890б-
&б#
!і
inputs         ђ
ф "і         ђЦ
D__inference_dense_23_layer_call_and_return_conditional_losses_937469] 0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ }
)__inference_dense_23_layer_call_fn_937458P 0б-
&б#
!і
inputs         ђ
ф "і         И
F__inference_dropout_21_layer_call_and_return_conditional_losses_937746n<б9
2б/
)і&
inputs         ђ
p 
ф ".б+
$і!
0         ђ
џ И
F__inference_dropout_21_layer_call_and_return_conditional_losses_937758n<б9
2б/
)і&
inputs         ђ
p
ф ".б+
$і!
0         ђ
џ љ
+__inference_dropout_21_layer_call_fn_937736a<б9
2б/
)і&
inputs         ђ
p 
ф "!і         ђљ
+__inference_dropout_21_layer_call_fn_937741a<б9
2б/
)і&
inputs         ђ
p
ф "!і         ђе
F__inference_dropout_22_layer_call_and_return_conditional_losses_937816^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_22_layer_call_and_return_conditional_losses_937828^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ђ
+__inference_dropout_22_layer_call_fn_937806Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђђ
+__inference_dropout_22_layer_call_fn_937811Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђе
F__inference_dropout_23_layer_call_and_return_conditional_losses_937875^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_23_layer_call_and_return_conditional_losses_937887^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ђ
+__inference_dropout_23_layer_call_fn_937865Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђђ
+__inference_dropout_23_layer_call_fn_937870Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђФ
E__inference_flatten_7_layer_call_and_return_conditional_losses_937769b8б5
.б+
)і&
inputs         ђ
ф "&б#
і
0         ђ 
џ Ѓ
*__inference_flatten_7_layer_call_fn_937763U8б5
.б+
)і&
inputs         ђ
ф "і         ђ И
D__inference_lambda_7_layer_call_and_return_conditional_losses_937487p?б<
5б2
(і%
inputs         KK

 
p 
ф "-б*
#і 
0         KK
џ И
D__inference_lambda_7_layer_call_and_return_conditional_losses_937495p?б<
5б2
(і%
inputs         KK

 
p
ф "-б*
#і 
0         KK
џ љ
)__inference_lambda_7_layer_call_fn_937474c?б<
5б2
(і%
inputs         KK

 
p 
ф " і         KKљ
)__inference_lambda_7_layer_call_fn_937479c?б<
5б2
(і%
inputs         KK

 
p
ф " і         KK;
__inference_loss_fn_0_937898,б

б 
ф "і ;
__inference_loss_fn_1_9379096б

б 
ф "і ;
__inference_loss_fn_2_9379208б

б 
ф "і №
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_934766ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_35_layer_call_fn_934772ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_934778ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_36_layer_call_fn_934784ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_934790ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_37_layer_call_fn_934796ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_934802ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_38_layer_call_fn_934808ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_934814ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_39_layer_call_fn_934820ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ╔
H__inference_sequential_7_layer_call_and_return_conditional_losses_937110}*+:;,-./0123456789?б<
5б2
(і%
inputs         KK
p 

 
ф "&б#
і
0         ђ
џ ╔
H__inference_sequential_7_layer_call_and_return_conditional_losses_937230}*+:;,-./0123456789?б<
5б2
(і%
inputs         KK
p

 
ф "&б#
і
0         ђ
џ м
H__inference_sequential_7_layer_call_and_return_conditional_losses_937329Ё*+:;,-./0123456789GбD
=б:
0і-
lambda_7_input         KK
p 

 
ф "&б#
і
0         ђ
џ м
H__inference_sequential_7_layer_call_and_return_conditional_losses_937449Ё*+:;,-./0123456789GбD
=б:
0і-
lambda_7_input         KK
p

 
ф "&б#
і
0         ђ
џ Е
-__inference_sequential_7_layer_call_fn_936888x*+:;,-./0123456789GбD
=б:
0і-
lambda_7_input         KK
p 

 
ф "і         ђА
-__inference_sequential_7_layer_call_fn_936929p*+:;,-./0123456789?б<
5б2
(і%
inputs         KK
p 

 
ф "і         ђА
-__inference_sequential_7_layer_call_fn_936970p*+:;,-./0123456789?б<
5б2
(і%
inputs         KK
p

 
ф "і         ђЕ
-__inference_sequential_7_layer_call_fn_937011x*+:;,-./0123456789GбD
=б:
0і-
lambda_7_input         KK
p

 
ф "і         ђ╣
$__inference_signature_wrapper_936183љ*+:;,-./0123456789 Cб@
б 
9ф6
4
input_1)і&
input_1         KK"3ф0
.
output_1"і
output_1         