 Ј$
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
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ* 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	ђ*
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
ј
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma
Є
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
ї
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
Ё
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
ё
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
: *
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
: *
dtype0
Ё
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*!
shared_nameconv2d_21/kernel
~
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*'
_output_shapes
: ђ*
dtype0
u
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_21/bias
n
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_22/kernel

$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_22/bias
n
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_23/kernel

$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_23/bias
n
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_24/kernel

$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_24/bias
n
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes	
:ђ*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
ђђ*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:ђ*
dtype0
|
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ* 
shared_namedense_13/kernel
u
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel* 
_output_shapes
:
ђђ*
dtype0
s
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
:ђ*
dtype0
џ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
Њ
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
б
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
Џ
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
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
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_14/kernel/m
ѓ
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes
:	ђ*
dtype0
ђ
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
ю
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/m
Ћ
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/m
Њ
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_20/kernel/m
І
+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_20/bias/m
{
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
_output_shapes
: *
dtype0
Њ
Adam/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*(
shared_nameAdam/conv2d_21/kernel/m
ї
+Adam/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/m*'
_output_shapes
: ђ*
dtype0
Ѓ
Adam/conv2d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_21/bias/m
|
)Adam/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_22/kernel/m
Ї
+Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_22/bias/m
|
)Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_23/kernel/m
Ї
+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_23/bias/m
|
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_24/kernel/m
Ї
+Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_24/bias/m
|
)Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/m*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_12/kernel/m
Ѓ
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_13/kernel/m
Ѓ
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_13/bias/m
z
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_14/kernel/v
ѓ
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes
:	ђ*
dtype0
ђ
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
ю
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/v
Ћ
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/v
Њ
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_20/kernel/v
І
+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_20/bias/v
{
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
_output_shapes
: *
dtype0
Њ
Adam/conv2d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*(
shared_nameAdam/conv2d_21/kernel/v
ї
+Adam/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/v*'
_output_shapes
: ђ*
dtype0
Ѓ
Adam/conv2d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_21/bias/v
|
)Adam/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_22/kernel/v
Ї
+Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_22/bias/v
|
)Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_23/kernel/v
Ї
+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_23/bias/v
|
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_24/kernel/v
Ї
+Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_24/bias/v
|
)Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/v*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_12/kernel/v
Ѓ
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_13/kernel/v
Ѓ
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_13/bias/v
z
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
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
regularization_losses
<layer_metrics
trainable_variables

=layers
>metrics
?non_trainable_variables
@layer_regularization_losses
	variables
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
regularization_losses
іlayer_metrics
trainable_variables
Іlayers
їmetrics
Їnon_trainable_variables
 јlayer_regularization_losses
	variables
NL
VARIABLE_VALUEdense_14/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_14/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
▓
!regularization_losses
Јlayers
"trainable_variables
љmetrics
Љnon_trainable_variables
#	variables
 њlayer_regularization_losses
Њlayer_metrics
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
VARIABLE_VALUEbatch_normalization_4/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_4/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_20/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_20/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_21/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_21/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_22/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_22/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_23/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_23/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_24/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_24/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_12/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_12/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_13/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_13/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_4/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_4/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

ћ0
Ћ1

:0
;1
 
 
 
 
▓
Aregularization_losses
ќlayers
Btrainable_variables
Ќmetrics
ўnon_trainable_variables
C	variables
 Ўlayer_regularization_losses
џlayer_metrics
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
Fregularization_losses
Џlayers
Gtrainable_variables
юmetrics
Юnon_trainable_variables
H	variables
 ъlayer_regularization_losses
Ъlayer_metrics
 

,0
-1

,0
-1
▓
Jregularization_losses
аlayers
Ktrainable_variables
Аmetrics
бnon_trainable_variables
L	variables
 Бlayer_regularization_losses
цlayer_metrics
 
 
 
▓
Nregularization_losses
Цlayers
Otrainable_variables
дmetrics
Дnon_trainable_variables
P	variables
 еlayer_regularization_losses
Еlayer_metrics
 

.0
/1

.0
/1
▓
Rregularization_losses
фlayers
Strainable_variables
Фmetrics
гnon_trainable_variables
T	variables
 Гlayer_regularization_losses
«layer_metrics
 
 
 
▓
Vregularization_losses
»layers
Wtrainable_variables
░metrics
▒non_trainable_variables
X	variables
 ▓layer_regularization_losses
│layer_metrics
 

00
11

00
11
▓
Zregularization_losses
┤layers
[trainable_variables
хmetrics
Хnon_trainable_variables
\	variables
 иlayer_regularization_losses
Иlayer_metrics
 
 
 
▓
^regularization_losses
╣layers
_trainable_variables
║metrics
╗non_trainable_variables
`	variables
 ╝layer_regularization_losses
йlayer_metrics
 

20
31

20
31
▓
bregularization_losses
Йlayers
ctrainable_variables
┐metrics
└non_trainable_variables
d	variables
 ┴layer_regularization_losses
┬layer_metrics
 
 
 
▓
fregularization_losses
├layers
gtrainable_variables
─metrics
┼non_trainable_variables
h	variables
 кlayer_regularization_losses
Кlayer_metrics
 

40
51

40
51
▓
jregularization_losses
╚layers
ktrainable_variables
╔metrics
╩non_trainable_variables
l	variables
 ╦layer_regularization_losses
╠layer_metrics
 
 
 
▓
nregularization_losses
═layers
otrainable_variables
╬metrics
¤non_trainable_variables
p	variables
 лlayer_regularization_losses
Лlayer_metrics
 
 
 
▓
rregularization_losses
мlayers
strainable_variables
Мmetrics
нnon_trainable_variables
t	variables
 Нlayer_regularization_losses
оlayer_metrics
 
 
 
▓
vregularization_losses
Оlayers
wtrainable_variables
пmetrics
┘non_trainable_variables
x	variables
 ┌layer_regularization_losses
█layer_metrics
 

60
71

60
71
▓
zregularization_losses
▄layers
{trainable_variables
Пmetrics
яnon_trainable_variables
|	variables
 ▀layer_regularization_losses
Яlayer_metrics
 
 
 
│
~regularization_losses
рlayers
trainable_variables
Рmetrics
сnon_trainable_variables
ђ	variables
 Сlayer_regularization_losses
тlayer_metrics
 

80
91

80
91
х
ѓregularization_losses
Тlayers
Ѓtrainable_variables
уmetrics
Уnon_trainable_variables
ё	variables
 жlayer_regularization_losses
Жlayer_metrics
 
 
 
х
єregularization_losses
вlayers
Єtrainable_variables
Вmetrics
ьnon_trainable_variables
ѕ	variables
 Ьlayer_regularization_losses
№layer_metrics
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
VARIABLE_VALUEAdam/dense_14/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_14/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_20/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_20/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_21/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_21/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_22/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_22/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_23/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_23/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_24/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_24/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_12/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_12/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_13/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_13/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_14/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_14/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_20/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_20/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_21/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_21/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_22/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_22/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_23/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_23/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_24/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_24/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_12/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_12/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_13/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_13/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
і
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
В
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias* 
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
$__inference_signature_wrapper_587622
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp+Adam/conv2d_21/kernel/m/Read/ReadVariableOp)Adam/conv2d_21/bias/m/Read/ReadVariableOp+Adam/conv2d_22/kernel/m/Read/ReadVariableOp)Adam/conv2d_22/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp+Adam/conv2d_21/kernel/v/Read/ReadVariableOp)Adam/conv2d_21/bias/v/Read/ReadVariableOp+Adam/conv2d_22/kernel/v/Read/ReadVariableOp)Adam/conv2d_22/bias/v/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpConst*N
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
__inference__traced_save_589577
л
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_4/gammabatch_normalization_4/betaconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancetotalcounttotal_1count_1Adam/dense_14/kernel/mAdam/dense_14/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/conv2d_21/kernel/mAdam/conv2d_21/bias/mAdam/conv2d_22/kernel/mAdam/conv2d_22/bias/mAdam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/conv2d_24/kernel/mAdam/conv2d_24/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/conv2d_21/kernel/vAdam/conv2d_21/bias/vAdam/conv2d_22/kernel/vAdam/conv2d_22/bias/vAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v*M
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
"__inference__traced_restore_589782ил
Л
б
*__inference_conv2d_24_layer_call_fn_589170

inputs#
unknown:ђђ
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
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_5863922
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

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
я
M
1__inference_max_pooling2d_20_layer_call_fn_586211

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
L__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_5862052
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
р
E
)__inference_lambda_4_layer_call_fn_588934

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
D__inference_lambda_4_layer_call_and_return_conditional_losses_5867372
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
Ш
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_586624

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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
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
:         ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
А
Ђ
E__inference_conv2d_23_layer_call_and_return_conditional_losses_586374

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
├
│
$__inference_signature_wrapper_587622
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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

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
!__inference__wrapped_model_5860732
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
У
│
__inference_loss_fn_1_589348N
:dense_12_kernel_regularizer_square_readvariableop_resource:
ђђ
identityѕб1dense_12/kernel/Regularizer/Square/ReadVariableOpс
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_12_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulџ
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
Б
Ќ
)__inference_dense_14_layer_call_fn_588908

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
D__inference_dense_14_layer_call_and_return_conditional_losses_5871462
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
џ
╗
__inference_loss_fn_0_589337U
;conv2d_20_kernel_regularizer_square_readvariableop_resource: 
identityѕб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpВ
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_20_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulю
IdentityIdentity$conv2d_20/kernel/Regularizer/mul:z:03^conv2d_20/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp
г
h
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_586217

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
╣
г
D__inference_dense_12_layer_call_and_return_conditional_losses_586431

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
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
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┬
`
D__inference_lambda_4_layer_call_and_return_conditional_losses_586274

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
┐
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_586139

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
ж
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_586412

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ю
ђ
E__inference_conv2d_21_layer_call_and_return_conditional_losses_586338

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
х
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_589257

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
Д
Ў
)__inference_dense_13_layer_call_fn_589299

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
D__inference_dense_13_layer_call_and_return_conditional_losses_5864612
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
г
h
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_586229

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
Ќ
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_586404

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_21_layer_call_fn_586223

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
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_5862172
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
E__inference_conv2d_24_layer_call_and_return_conditional_losses_586392

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
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
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

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
ј▓
 
H__inference_sequential_4_layer_call_and_return_conditional_losses_588724
lambda_4_input;
-batch_normalization_4_readvariableop_resource:=
/batch_normalization_4_readvariableop_1_resource:L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_20_conv2d_readvariableop_resource: 7
)conv2d_20_biasadd_readvariableop_resource: C
(conv2d_21_conv2d_readvariableop_resource: ђ8
)conv2d_21_biasadd_readvariableop_resource:	ђD
(conv2d_22_conv2d_readvariableop_resource:ђђ8
)conv2d_22_biasadd_readvariableop_resource:	ђD
(conv2d_23_conv2d_readvariableop_resource:ђђ8
)conv2d_23_biasadd_readvariableop_resource:	ђD
(conv2d_24_conv2d_readvariableop_resource:ђђ8
)conv2d_24_biasadd_readvariableop_resource:	ђ;
'dense_12_matmul_readvariableop_resource:
ђђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identityѕб$batch_normalization_4/AssignNewValueб&batch_normalization_4/AssignNewValue_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б conv2d_20/BiasAdd/ReadVariableOpбconv2d_20/Conv2D/ReadVariableOpб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб conv2d_21/BiasAdd/ReadVariableOpбconv2d_21/Conv2D/ReadVariableOpб conv2d_22/BiasAdd/ReadVariableOpбconv2d_22/Conv2D/ReadVariableOpб conv2d_23/BiasAdd/ReadVariableOpбconv2d_23/Conv2D/ReadVariableOpб conv2d_24/BiasAdd/ReadVariableOpбconv2d_24/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stackЎ
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1Ў
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2▓
lambda_4/strided_sliceStridedSlicelambda_4_input%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_4/strided_sliceХ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_4/FusedBatchNormV3░
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue╝
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1│
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_20/Conv2D/ReadVariableOpт
conv2d_20/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_20/Conv2Dф
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp░
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_20/Relu╩
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool┤
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_21/Conv2D/ReadVariableOpП
conv2d_21/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_21/Conv2DФ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp▒
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_21/BiasAdd
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_21/Relu╦
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPoolх
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_22/Conv2D/ReadVariableOpП
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_22/Conv2DФ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp▒
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_22/Relu╦
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPoolх
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_23/Conv2D/ReadVariableOpП
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
conv2d_23/Conv2DФ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp▒
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
conv2d_23/Relu╦
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPoolх
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_24/Conv2D/ReadVariableOpП
conv2d_24/Conv2DConv2D!max_pooling2d_23/MaxPool:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_24/Conv2DФ
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp▒
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_24/BiasAdd
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_24/Relu╦
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_24/MaxPooly
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_12/dropout/ConstИ
dropout_12/dropout/MulMul!max_pooling2d_24/MaxPool:output:0!dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout_12/dropout/MulЁ
dropout_12/dropout/ShapeShape!max_pooling2d_24/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shapeя
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype021
/dropout_12/dropout/random_uniform/RandomUniformІ
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_12/dropout/GreaterEqual/yз
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2!
dropout_12/dropout/GreaterEqualЕ
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout_12/dropout/Cast»
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout_12/dropout/Mul_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_4/Constю
flatten_4/ReshapeReshapedropout_12/dropout/Mul_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_4/Reshapeф
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/MatMulе
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_12/BiasAdd/ReadVariableOpд
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_12/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Constф
dropout_13/dropout/MulMuldense_12/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shapeо
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_13/dropout/random_uniform/RandomUniformІ
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/yв
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_13/dropout/GreaterEqualА
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_13/dropout/CastД
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_13/dropout/Mul_1ф
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_13/MatMul/ReadVariableOpЦ
dense_13/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/MatMulе
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_13/BiasAdd/ReadVariableOpд
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_13/Reluy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_14/dropout/Constф
dropout_14/dropout/MulMuldense_13/Relu:activations:0!dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shapeо
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_14/dropout/random_uniform/RandomUniformІ
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_14/dropout/GreaterEqual/yв
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_14/dropout/GreaterEqualА
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_14/dropout/CastД
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_14/dropout/Mul_1┘
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulл
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulл
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul 
IdentityIdentitydropout_14/dropout/Mul_1:z:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_4_input
Џ
о
!__inference__wrapped_model_586073
input_1
cnn3_586031:
cnn3_586033:
cnn3_586035:
cnn3_586037:%
cnn3_586039: 
cnn3_586041: &
cnn3_586043: ђ
cnn3_586045:	ђ'
cnn3_586047:ђђ
cnn3_586049:	ђ'
cnn3_586051:ђђ
cnn3_586053:	ђ'
cnn3_586055:ђђ
cnn3_586057:	ђ
cnn3_586059:
ђђ
cnn3_586061:	ђ
cnn3_586063:
ђђ
cnn3_586065:	ђ
cnn3_586067:	ђ
cnn3_586069:
identityѕбCNN3/StatefulPartitionedCallв
CNN3/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn3_586031cnn3_586033cnn3_586035cnn3_586037cnn3_586039cnn3_586041cnn3_586043cnn3_586045cnn3_586047cnn3_586049cnn3_586051cnn3_586053cnn3_586055cnn3_586057cnn3_586059cnn3_586061cnn3_586063cnn3_586065cnn3_586067cnn3_586069* 
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
__inference_call_5301522
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
ц
Л
6__inference_batch_normalization_4_layer_call_fn_589058

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5867102
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
╩
Ъ
*__inference_conv2d_20_layer_call_fn_589090

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
E__inference_conv2d_20_layer_call_and_return_conditional_losses_5863202
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
╣
г
D__inference_dense_12_layer_call_and_return_conditional_losses_589231

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
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
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш▒
э
H__inference_sequential_4_layer_call_and_return_conditional_losses_588505

inputs;
-batch_normalization_4_readvariableop_resource:=
/batch_normalization_4_readvariableop_1_resource:L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_20_conv2d_readvariableop_resource: 7
)conv2d_20_biasadd_readvariableop_resource: C
(conv2d_21_conv2d_readvariableop_resource: ђ8
)conv2d_21_biasadd_readvariableop_resource:	ђD
(conv2d_22_conv2d_readvariableop_resource:ђђ8
)conv2d_22_biasadd_readvariableop_resource:	ђD
(conv2d_23_conv2d_readvariableop_resource:ђђ8
)conv2d_23_biasadd_readvariableop_resource:	ђD
(conv2d_24_conv2d_readvariableop_resource:ђђ8
)conv2d_24_biasadd_readvariableop_resource:	ђ;
'dense_12_matmul_readvariableop_resource:
ђђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identityѕб$batch_normalization_4/AssignNewValueб&batch_normalization_4/AssignNewValue_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б conv2d_20/BiasAdd/ReadVariableOpбconv2d_20/Conv2D/ReadVariableOpб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб conv2d_21/BiasAdd/ReadVariableOpбconv2d_21/Conv2D/ReadVariableOpб conv2d_22/BiasAdd/ReadVariableOpбconv2d_22/Conv2D/ReadVariableOpб conv2d_23/BiasAdd/ReadVariableOpбconv2d_23/Conv2D/ReadVariableOpб conv2d_24/BiasAdd/ReadVariableOpбconv2d_24/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stackЎ
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1Ў
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2ф
lambda_4/strided_sliceStridedSliceinputs%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_4/strided_sliceХ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_4/FusedBatchNormV3░
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue╝
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1│
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_20/Conv2D/ReadVariableOpт
conv2d_20/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_20/Conv2Dф
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp░
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_20/Relu╩
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool┤
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_21/Conv2D/ReadVariableOpП
conv2d_21/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_21/Conv2DФ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp▒
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_21/BiasAdd
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_21/Relu╦
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPoolх
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_22/Conv2D/ReadVariableOpП
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_22/Conv2DФ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp▒
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_22/Relu╦
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPoolх
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_23/Conv2D/ReadVariableOpП
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
conv2d_23/Conv2DФ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp▒
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
conv2d_23/Relu╦
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPoolх
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_24/Conv2D/ReadVariableOpП
conv2d_24/Conv2DConv2D!max_pooling2d_23/MaxPool:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_24/Conv2DФ
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp▒
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_24/BiasAdd
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_24/Relu╦
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_24/MaxPooly
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_12/dropout/ConstИ
dropout_12/dropout/MulMul!max_pooling2d_24/MaxPool:output:0!dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout_12/dropout/MulЁ
dropout_12/dropout/ShapeShape!max_pooling2d_24/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shapeя
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype021
/dropout_12/dropout/random_uniform/RandomUniformІ
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_12/dropout/GreaterEqual/yз
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2!
dropout_12/dropout/GreaterEqualЕ
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout_12/dropout/Cast»
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout_12/dropout/Mul_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_4/Constю
flatten_4/ReshapeReshapedropout_12/dropout/Mul_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_4/Reshapeф
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/MatMulе
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_12/BiasAdd/ReadVariableOpд
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_12/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Constф
dropout_13/dropout/MulMuldense_12/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shapeо
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_13/dropout/random_uniform/RandomUniformІ
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/yв
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_13/dropout/GreaterEqualА
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_13/dropout/CastД
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_13/dropout/Mul_1ф
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_13/MatMul/ReadVariableOpЦ
dense_13/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/MatMulе
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_13/BiasAdd/ReadVariableOpд
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_13/Reluy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_14/dropout/Constф
dropout_14/dropout/MulMuldense_13/Relu:activations:0!dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shapeо
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_14/dropout/random_uniform/RandomUniformІ
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_14/dropout/GreaterEqual/yв
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_14/dropout/GreaterEqualА
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_14/dropout/CastД
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_14/dropout/Mul_1┘
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulл
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulл
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul 
IdentityIdentitydropout_14/dropout/Mul_1:z:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┬▀
ћ
@__inference_CNN3_layer_call_and_return_conditional_losses_587855

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource: D
6sequential_4_conv2d_20_biasadd_readvariableop_resource: P
5sequential_4_conv2d_21_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_21_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_22_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_22_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_23_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_23_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_24_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_12_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpб1sequential_4/batch_normalization_4/AssignNewValueб3sequential_4/batch_normalization_4/AssignNewValue_1бBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_20/BiasAdd/ReadVariableOpб,sequential_4/conv2d_20/Conv2D/ReadVariableOpб-sequential_4/conv2d_21/BiasAdd/ReadVariableOpб,sequential_4/conv2d_21/Conv2D/ReadVariableOpб-sequential_4/conv2d_22/BiasAdd/ReadVariableOpб,sequential_4/conv2d_22/Conv2D/ReadVariableOpб-sequential_4/conv2d_23/BiasAdd/ReadVariableOpб,sequential_4/conv2d_23/Conv2D/ReadVariableOpб-sequential_4/conv2d_24/BiasAdd/ReadVariableOpб,sequential_4/conv2d_24/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2в
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1л
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<25
3sequential_4/batch_normalization_4/FusedBatchNormV3ы
1sequential_4/batch_normalization_4/AssignNewValueAssignVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource@sequential_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_4/batch_normalization_4/AssignNewValue§
3sequential_4/batch_normalization_4/AssignNewValue_1AssignVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceDsequential_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0E^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_4/batch_normalization_4/AssignNewValue_1┌
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_20/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_20/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_20/Conv2DЛ
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpС
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_20/BiasAddЦ
sequential_4/conv2d_20/ReluRelu'sequential_4/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_20/Reluы
%sequential_4/max_pooling2d_20/MaxPoolMaxPool)sequential_4/conv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_20/MaxPool█
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_21/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_21/Conv2DConv2D.sequential_4/max_pooling2d_20/MaxPool:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_21/Conv2Dм
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpт
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_21/BiasAddд
sequential_4/conv2d_21/ReluRelu'sequential_4/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_21/ReluЫ
%sequential_4/max_pooling2d_21/MaxPoolMaxPool)sequential_4/conv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_21/MaxPool▄
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_22/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_22/Conv2DConv2D.sequential_4/max_pooling2d_21/MaxPool:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_22/Conv2Dм
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpт
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_22/BiasAddд
sequential_4/conv2d_22/ReluRelu'sequential_4/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_22/ReluЫ
%sequential_4/max_pooling2d_22/MaxPoolMaxPool)sequential_4/conv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_22/MaxPool▄
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_23/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_23/Conv2DConv2D.sequential_4/max_pooling2d_22/MaxPool:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_4/conv2d_23/Conv2Dм
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpт
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_4/conv2d_23/BiasAddд
sequential_4/conv2d_23/ReluRelu'sequential_4/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_4/conv2d_23/ReluЫ
%sequential_4/max_pooling2d_23/MaxPoolMaxPool)sequential_4/conv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_23/MaxPool▄
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_24/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_24/Conv2DConv2D.sequential_4/max_pooling2d_23/MaxPool:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_24/Conv2Dм
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpт
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_24/BiasAddд
sequential_4/conv2d_24/ReluRelu'sequential_4/conv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_24/ReluЫ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool)sequential_4/conv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_24/MaxPoolЊ
%sequential_4/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2'
%sequential_4/dropout_12/dropout/ConstВ
#sequential_4/dropout_12/dropout/MulMul.sequential_4/max_pooling2d_24/MaxPool:output:0.sequential_4/dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2%
#sequential_4/dropout_12/dropout/Mulг
%sequential_4/dropout_12/dropout/ShapeShape.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_12/dropout/ShapeЁ
<sequential_4/dropout_12/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_12/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_4/dropout_12/dropout/GreaterEqual/yД
,sequential_4/dropout_12/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_12/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2.
,sequential_4/dropout_12/dropout/GreaterEqualл
$sequential_4/dropout_12/dropout/CastCast0sequential_4/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2&
$sequential_4/dropout_12/dropout/Castс
%sequential_4/dropout_12/dropout/Mul_1Mul'sequential_4/dropout_12/dropout/Mul:z:0(sequential_4/dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2'
%sequential_4/dropout_12/dropout/Mul_1Ї
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_4/flatten_4/Constл
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/dropout/Mul_1:z:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2 
sequential_4/flatten_4/ReshapeЛ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluЊ
%sequential_4/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_4/dropout_13/dropout/Constя
#sequential_4/dropout_13/dropout/MulMul(sequential_4/dense_12/Relu:activations:0.sequential_4/dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_4/dropout_13/dropout/Mulд
%sequential_4/dropout_13/dropout/ShapeShape(sequential_4/dense_12/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_13/dropout/Shape§
<sequential_4/dropout_13/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_13/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_4/dropout_13/dropout/GreaterEqual/yЪ
,sequential_4/dropout_13/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_13/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_4/dropout_13/dropout/GreaterEqual╚
$sequential_4/dropout_13/dropout/CastCast0sequential_4/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_4/dropout_13/dropout/Cast█
%sequential_4/dropout_13/dropout/Mul_1Mul'sequential_4/dropout_13/dropout/Mul:z:0(sequential_4/dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_4/dropout_13/dropout/Mul_1Л
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/dropout/Mul_1:z:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluЊ
%sequential_4/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_4/dropout_14/dropout/Constя
#sequential_4/dropout_14/dropout/MulMul(sequential_4/dense_13/Relu:activations:0.sequential_4/dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_4/dropout_14/dropout/Mulд
%sequential_4/dropout_14/dropout/ShapeShape(sequential_4/dense_13/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_14/dropout/Shape§
<sequential_4/dropout_14/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_14/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_4/dropout_14/dropout/GreaterEqual/yЪ
,sequential_4/dropout_14/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_14/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_4/dropout_14/dropout/GreaterEqual╚
$sequential_4/dropout_14/dropout/CastCast0sequential_4/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_4/dropout_14/dropout/Cast█
%sequential_4/dropout_14/dropout/Mul_1Mul'sequential_4/dropout_14/dropout/Mul:z:0(sequential_4/dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_4/dropout_14/dropout/Mul_1Е
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/SoftmaxТ
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulП
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulП
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul├

IdentityIdentitydense_14/Softmax:softmax:03^conv2d_20/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^sequential_4/batch_normalization_4/AssignNewValue4^sequential_4/batch_normalization_4/AssignNewValue_1C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1sequential_4/batch_normalization_4/AssignNewValue1sequential_4/batch_normalization_4/AssignNewValue2j
3sequential_4/batch_normalization_4/AssignNewValue_13sequential_4/batch_normalization_4/AssignNewValue_12ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
У
│
__inference_loss_fn_2_589359N
:dense_13_kernel_regularizer_square_readvariableop_resource:
ђђ
identityѕб1dense_13/kernel/Regularizer/Square/ReadVariableOpс
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_13_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulџ
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
▒Њ
С
__inference_call_530152

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource: D
6sequential_4_conv2d_20_biasadd_readvariableop_resource: P
5sequential_4_conv2d_21_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_21_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_22_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_22_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_23_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_23_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_24_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_12_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_20/BiasAdd/ReadVariableOpб,sequential_4/conv2d_20/Conv2D/ReadVariableOpб-sequential_4/conv2d_21/BiasAdd/ReadVariableOpб,sequential_4/conv2d_21/Conv2D/ReadVariableOpб-sequential_4/conv2d_22/BiasAdd/ReadVariableOpб,sequential_4/conv2d_22/Conv2D/ReadVariableOpб-sequential_4/conv2d_23/BiasAdd/ReadVariableOpб,sequential_4/conv2d_23/Conv2D/ReadVariableOpб-sequential_4/conv2d_24/BiasAdd/ReadVariableOpб,sequential_4/conv2d_24/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2в
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_20/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_20/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_20/Conv2DЛ
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpС
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_20/BiasAddЦ
sequential_4/conv2d_20/ReluRelu'sequential_4/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_20/Reluы
%sequential_4/max_pooling2d_20/MaxPoolMaxPool)sequential_4/conv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_20/MaxPool█
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_21/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_21/Conv2DConv2D.sequential_4/max_pooling2d_20/MaxPool:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_21/Conv2Dм
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpт
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_21/BiasAddд
sequential_4/conv2d_21/ReluRelu'sequential_4/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_21/ReluЫ
%sequential_4/max_pooling2d_21/MaxPoolMaxPool)sequential_4/conv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_21/MaxPool▄
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_22/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_22/Conv2DConv2D.sequential_4/max_pooling2d_21/MaxPool:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_22/Conv2Dм
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpт
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_22/BiasAddд
sequential_4/conv2d_22/ReluRelu'sequential_4/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_22/ReluЫ
%sequential_4/max_pooling2d_22/MaxPoolMaxPool)sequential_4/conv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_22/MaxPool▄
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_23/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_23/Conv2DConv2D.sequential_4/max_pooling2d_22/MaxPool:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_4/conv2d_23/Conv2Dм
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpт
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_4/conv2d_23/BiasAddд
sequential_4/conv2d_23/ReluRelu'sequential_4/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_4/conv2d_23/ReluЫ
%sequential_4/max_pooling2d_23/MaxPoolMaxPool)sequential_4/conv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_23/MaxPool▄
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_24/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_24/Conv2DConv2D.sequential_4/max_pooling2d_23/MaxPool:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_24/Conv2Dм
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpт
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_24/BiasAddд
sequential_4/conv2d_24/ReluRelu'sequential_4/conv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_24/ReluЫ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool)sequential_4/conv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_24/MaxPool╗
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_4/flatten_4/Constл
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2 
sequential_4/flatten_4/ReshapeЛ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluГ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluГ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/Softmax╝
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
яЇ
»
H__inference_sequential_4_layer_call_and_return_conditional_losses_588604
lambda_4_input;
-batch_normalization_4_readvariableop_resource:=
/batch_normalization_4_readvariableop_1_resource:L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_20_conv2d_readvariableop_resource: 7
)conv2d_20_biasadd_readvariableop_resource: C
(conv2d_21_conv2d_readvariableop_resource: ђ8
)conv2d_21_biasadd_readvariableop_resource:	ђD
(conv2d_22_conv2d_readvariableop_resource:ђђ8
)conv2d_22_biasadd_readvariableop_resource:	ђD
(conv2d_23_conv2d_readvariableop_resource:ђђ8
)conv2d_23_biasadd_readvariableop_resource:	ђD
(conv2d_24_conv2d_readvariableop_resource:ђђ8
)conv2d_24_biasadd_readvariableop_resource:	ђ;
'dense_12_matmul_readvariableop_resource:
ђђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identityѕб5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б conv2d_20/BiasAdd/ReadVariableOpбconv2d_20/Conv2D/ReadVariableOpб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб conv2d_21/BiasAdd/ReadVariableOpбconv2d_21/Conv2D/ReadVariableOpб conv2d_22/BiasAdd/ReadVariableOpбconv2d_22/Conv2D/ReadVariableOpб conv2d_23/BiasAdd/ReadVariableOpбconv2d_23/Conv2D/ReadVariableOpб conv2d_24/BiasAdd/ReadVariableOpбconv2d_24/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stackЎ
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1Ў
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2▓
lambda_4/strided_sliceStridedSlicelambda_4_input%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_4/strided_sliceХ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1у
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3│
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_20/Conv2D/ReadVariableOpт
conv2d_20/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_20/Conv2Dф
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp░
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_20/Relu╩
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool┤
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_21/Conv2D/ReadVariableOpП
conv2d_21/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_21/Conv2DФ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp▒
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_21/BiasAdd
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_21/Relu╦
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPoolх
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_22/Conv2D/ReadVariableOpП
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_22/Conv2DФ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp▒
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_22/Relu╦
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPoolх
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_23/Conv2D/ReadVariableOpП
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
conv2d_23/Conv2DФ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp▒
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
conv2d_23/Relu╦
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPoolх
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_24/Conv2D/ReadVariableOpП
conv2d_24/Conv2DConv2D!max_pooling2d_23/MaxPool:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_24/Conv2DФ
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp▒
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_24/BiasAdd
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_24/Relu╦
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_24/MaxPoolћ
dropout_12/IdentityIdentity!max_pooling2d_24/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2
dropout_12/Identitys
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_4/Constю
flatten_4/ReshapeReshapedropout_12/Identity:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_4/Reshapeф
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/MatMulе
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_12/BiasAdd/ReadVariableOpд
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_12/Reluє
dropout_13/IdentityIdentitydense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_13/Identityф
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_13/MatMul/ReadVariableOpЦ
dense_13/MatMulMatMuldropout_13/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/MatMulе
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_13/BiasAdd/ReadVariableOpд
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_13/Reluє
dropout_14/IdentityIdentitydense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_14/Identity┘
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulл
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulл
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul»
IdentityIdentitydropout_14/Identity:output:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_4_input
Н
d
+__inference_dropout_13_layer_call_fn_589267

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
F__inference_dropout_13_layer_call_and_return_conditional_losses_5865852
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
Я
│
%__inference_CNN3_layer_call_fn_588178

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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

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
@__inference_CNN3_layer_call_and_return_conditional_losses_5871712
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
я
M
1__inference_max_pooling2d_22_layer_call_fn_586235

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
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_5862292
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
р
┤
%__inference_CNN3_layer_call_fn_588268
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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

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
@__inference_CNN3_layer_call_and_return_conditional_losses_5873352
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
р
E
)__inference_lambda_4_layer_call_fn_588929

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
D__inference_lambda_4_layer_call_and_return_conditional_losses_5862742
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
д
Л
6__inference_batch_normalization_4_layer_call_fn_589045

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5862932
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
І
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588952

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
Бm
№	
H__inference_sequential_4_layer_call_and_return_conditional_losses_586493

inputs*
batch_normalization_4_586294:*
batch_normalization_4_586296:*
batch_normalization_4_586298:*
batch_normalization_4_586300:*
conv2d_20_586321: 
conv2d_20_586323: +
conv2d_21_586339: ђ
conv2d_21_586341:	ђ,
conv2d_22_586357:ђђ
conv2d_22_586359:	ђ,
conv2d_23_586375:ђђ
conv2d_23_586377:	ђ,
conv2d_24_586393:ђђ
conv2d_24_586395:	ђ#
dense_12_586432:
ђђ
dense_12_586434:	ђ#
dense_13_586462:
ђђ
dense_13_586464:	ђ
identityѕб-batch_normalization_4/StatefulPartitionedCallб!conv2d_20/StatefulPartitionedCallб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб!conv2d_21/StatefulPartitionedCallб!conv2d_22/StatefulPartitionedCallб!conv2d_23/StatefulPartitionedCallб!conv2d_24/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб1dense_12/kernel/Regularizer/Square/ReadVariableOpб dense_13/StatefulPartitionedCallб1dense_13/kernel/Regularizer/Square/ReadVariableOpр
lambda_4/PartitionedCallPartitionedCallinputs*
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
D__inference_lambda_4_layer_call_and_return_conditional_losses_5862742
lambda_4/PartitionedCallй
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0batch_normalization_4_586294batch_normalization_4_586296batch_normalization_4_586298batch_normalization_4_586300*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5862932/
-batch_normalization_4/StatefulPartitionedCallо
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_20_586321conv2d_20_586323*
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
E__inference_conv2d_20_layer_call_and_return_conditional_losses_5863202#
!conv2d_20/StatefulPartitionedCallЮ
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_5862052"
 max_pooling2d_20/PartitionedCall╩
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0conv2d_21_586339conv2d_21_586341*
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
E__inference_conv2d_21_layer_call_and_return_conditional_losses_5863382#
!conv2d_21/StatefulPartitionedCallъ
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_5862172"
 max_pooling2d_21/PartitionedCall╩
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_22_586357conv2d_22_586359*
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
E__inference_conv2d_22_layer_call_and_return_conditional_losses_5863562#
!conv2d_22/StatefulPartitionedCallъ
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_5862292"
 max_pooling2d_22/PartitionedCall╩
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_586375conv2d_23_586377*
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
E__inference_conv2d_23_layer_call_and_return_conditional_losses_5863742#
!conv2d_23/StatefulPartitionedCallъ
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_5862412"
 max_pooling2d_23/PartitionedCall╩
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0conv2d_24_586393conv2d_24_586395*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_5863922#
!conv2d_24/StatefulPartitionedCallъ
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_5862532"
 max_pooling2d_24/PartitionedCallІ
dropout_12/PartitionedCallPartitionedCall)max_pooling2d_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5864042
dropout_12/PartitionedCallЩ
flatten_4/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_5864122
flatten_4/PartitionedCallХ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_586432dense_12_586434*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_5864312"
 dense_12/StatefulPartitionedCallЃ
dropout_13/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
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
F__inference_dropout_13_layer_call_and_return_conditional_losses_5864422
dropout_13/PartitionedCallи
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_13_586462dense_13_586464*
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
D__inference_dense_13_layer_call_and_return_conditional_losses_5864612"
 dense_13/StatefulPartitionedCallЃ
dropout_14/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
F__inference_dropout_14_layer_call_and_return_conditional_losses_5864722
dropout_14/PartitionedCall┴
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_586321*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulИ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_586432* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulИ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_586462* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul┐
IdentityIdentity#dropout_14/PartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┐
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588970

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
▒Њ
С
__inference_call_532654

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource: D
6sequential_4_conv2d_20_biasadd_readvariableop_resource: P
5sequential_4_conv2d_21_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_21_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_22_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_22_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_23_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_23_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_24_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_12_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_20/BiasAdd/ReadVariableOpб,sequential_4/conv2d_20/Conv2D/ReadVariableOpб-sequential_4/conv2d_21/BiasAdd/ReadVariableOpб,sequential_4/conv2d_21/Conv2D/ReadVariableOpб-sequential_4/conv2d_22/BiasAdd/ReadVariableOpб,sequential_4/conv2d_22/Conv2D/ReadVariableOpб-sequential_4/conv2d_23/BiasAdd/ReadVariableOpб,sequential_4/conv2d_23/Conv2D/ReadVariableOpб-sequential_4/conv2d_24/BiasAdd/ReadVariableOpб,sequential_4/conv2d_24/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2в
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_20/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_20/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_20/Conv2DЛ
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpС
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_20/BiasAddЦ
sequential_4/conv2d_20/ReluRelu'sequential_4/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_20/Reluы
%sequential_4/max_pooling2d_20/MaxPoolMaxPool)sequential_4/conv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_20/MaxPool█
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_21/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_21/Conv2DConv2D.sequential_4/max_pooling2d_20/MaxPool:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_21/Conv2Dм
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpт
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_21/BiasAddд
sequential_4/conv2d_21/ReluRelu'sequential_4/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_21/ReluЫ
%sequential_4/max_pooling2d_21/MaxPoolMaxPool)sequential_4/conv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_21/MaxPool▄
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_22/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_22/Conv2DConv2D.sequential_4/max_pooling2d_21/MaxPool:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_22/Conv2Dм
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpт
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_22/BiasAddд
sequential_4/conv2d_22/ReluRelu'sequential_4/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_22/ReluЫ
%sequential_4/max_pooling2d_22/MaxPoolMaxPool)sequential_4/conv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_22/MaxPool▄
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_23/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_23/Conv2DConv2D.sequential_4/max_pooling2d_22/MaxPool:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_4/conv2d_23/Conv2Dм
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpт
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_4/conv2d_23/BiasAddд
sequential_4/conv2d_23/ReluRelu'sequential_4/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_4/conv2d_23/ReluЫ
%sequential_4/max_pooling2d_23/MaxPoolMaxPool)sequential_4/conv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_23/MaxPool▄
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_24/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_24/Conv2DConv2D.sequential_4/max_pooling2d_23/MaxPool:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_24/Conv2Dм
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpт
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_24/BiasAddд
sequential_4/conv2d_24/ReluRelu'sequential_4/conv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_24/ReluЫ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool)sequential_4/conv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_24/MaxPool╗
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_4/flatten_4/Constл
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2 
sequential_4/flatten_4/ReshapeЛ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluГ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluГ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/Softmax╝
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ж
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_589203

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ь
Л
6__inference_batch_normalization_4_layer_call_fn_589019

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5860952
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
Єr
я

H__inference_sequential_4_layer_call_and_return_conditional_losses_586859

inputs*
batch_normalization_4_586787:*
batch_normalization_4_586789:*
batch_normalization_4_586791:*
batch_normalization_4_586793:*
conv2d_20_586796: 
conv2d_20_586798: +
conv2d_21_586802: ђ
conv2d_21_586804:	ђ,
conv2d_22_586808:ђђ
conv2d_22_586810:	ђ,
conv2d_23_586814:ђђ
conv2d_23_586816:	ђ,
conv2d_24_586820:ђђ
conv2d_24_586822:	ђ#
dense_12_586828:
ђђ
dense_12_586830:	ђ#
dense_13_586834:
ђђ
dense_13_586836:	ђ
identityѕб-batch_normalization_4/StatefulPartitionedCallб!conv2d_20/StatefulPartitionedCallб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб!conv2d_21/StatefulPartitionedCallб!conv2d_22/StatefulPartitionedCallб!conv2d_23/StatefulPartitionedCallб!conv2d_24/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб1dense_12/kernel/Regularizer/Square/ReadVariableOpб dense_13/StatefulPartitionedCallб1dense_13/kernel/Regularizer/Square/ReadVariableOpб"dropout_12/StatefulPartitionedCallб"dropout_13/StatefulPartitionedCallб"dropout_14/StatefulPartitionedCallр
lambda_4/PartitionedCallPartitionedCallinputs*
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
D__inference_lambda_4_layer_call_and_return_conditional_losses_5867372
lambda_4/PartitionedCall╗
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0batch_normalization_4_586787batch_normalization_4_586789batch_normalization_4_586791batch_normalization_4_586793*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5867102/
-batch_normalization_4/StatefulPartitionedCallо
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_20_586796conv2d_20_586798*
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
E__inference_conv2d_20_layer_call_and_return_conditional_losses_5863202#
!conv2d_20/StatefulPartitionedCallЮ
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_5862052"
 max_pooling2d_20/PartitionedCall╩
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0conv2d_21_586802conv2d_21_586804*
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
E__inference_conv2d_21_layer_call_and_return_conditional_losses_5863382#
!conv2d_21/StatefulPartitionedCallъ
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_5862172"
 max_pooling2d_21/PartitionedCall╩
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_22_586808conv2d_22_586810*
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
E__inference_conv2d_22_layer_call_and_return_conditional_losses_5863562#
!conv2d_22/StatefulPartitionedCallъ
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_5862292"
 max_pooling2d_22/PartitionedCall╩
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_586814conv2d_23_586816*
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
E__inference_conv2d_23_layer_call_and_return_conditional_losses_5863742#
!conv2d_23/StatefulPartitionedCallъ
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_5862412"
 max_pooling2d_23/PartitionedCall╩
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0conv2d_24_586820conv2d_24_586822*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_5863922#
!conv2d_24/StatefulPartitionedCallъ
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_5862532"
 max_pooling2d_24/PartitionedCallБ
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5866242$
"dropout_12/StatefulPartitionedCallѓ
flatten_4/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_5864122
flatten_4/PartitionedCallХ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_586828dense_12_586830*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_5864312"
 dense_12/StatefulPartitionedCall└
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
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
F__inference_dropout_13_layer_call_and_return_conditional_losses_5865852$
"dropout_13/StatefulPartitionedCall┐
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_13_586834dense_13_586836*
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
D__inference_dense_13_layer_call_and_return_conditional_losses_5864612"
 dense_13/StatefulPartitionedCall└
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*
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
F__inference_dropout_14_layer_call_and_return_conditional_losses_5865522$
"dropout_14/StatefulPartitionedCall┴
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_20_586796*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulИ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_586828* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulИ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_586834* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulХ
IdentityIdentity+dropout_14/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Л
б
*__inference_conv2d_23_layer_call_fn_589150

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
E__inference_conv2d_23_layer_call_and_return_conditional_losses_5863742
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
э
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_589245

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
╠▓
Ф
@__inference_CNN3_layer_call_and_return_conditional_losses_587961
input_1H
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource: D
6sequential_4_conv2d_20_biasadd_readvariableop_resource: P
5sequential_4_conv2d_21_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_21_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_22_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_22_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_23_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_23_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_24_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_12_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_20/BiasAdd/ReadVariableOpб,sequential_4/conv2d_20/Conv2D/ReadVariableOpб-sequential_4/conv2d_21/BiasAdd/ReadVariableOpб,sequential_4/conv2d_21/Conv2D/ReadVariableOpб-sequential_4/conv2d_22/BiasAdd/ReadVariableOpб,sequential_4/conv2d_22/Conv2D/ReadVariableOpб-sequential_4/conv2d_23/BiasAdd/ReadVariableOpб,sequential_4/conv2d_23/Conv2D/ReadVariableOpб-sequential_4/conv2d_24/BiasAdd/ReadVariableOpб,sequential_4/conv2d_24/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2В
#sequential_4/lambda_4/strided_sliceStridedSliceinput_12sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_20/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_20/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_20/Conv2DЛ
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpС
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_20/BiasAddЦ
sequential_4/conv2d_20/ReluRelu'sequential_4/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_20/Reluы
%sequential_4/max_pooling2d_20/MaxPoolMaxPool)sequential_4/conv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_20/MaxPool█
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_21/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_21/Conv2DConv2D.sequential_4/max_pooling2d_20/MaxPool:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_21/Conv2Dм
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpт
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_21/BiasAddд
sequential_4/conv2d_21/ReluRelu'sequential_4/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_21/ReluЫ
%sequential_4/max_pooling2d_21/MaxPoolMaxPool)sequential_4/conv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_21/MaxPool▄
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_22/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_22/Conv2DConv2D.sequential_4/max_pooling2d_21/MaxPool:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_22/Conv2Dм
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpт
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_22/BiasAddд
sequential_4/conv2d_22/ReluRelu'sequential_4/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_22/ReluЫ
%sequential_4/max_pooling2d_22/MaxPoolMaxPool)sequential_4/conv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_22/MaxPool▄
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_23/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_23/Conv2DConv2D.sequential_4/max_pooling2d_22/MaxPool:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_4/conv2d_23/Conv2Dм
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpт
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_4/conv2d_23/BiasAddд
sequential_4/conv2d_23/ReluRelu'sequential_4/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_4/conv2d_23/ReluЫ
%sequential_4/max_pooling2d_23/MaxPoolMaxPool)sequential_4/conv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_23/MaxPool▄
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_24/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_24/Conv2DConv2D.sequential_4/max_pooling2d_23/MaxPool:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_24/Conv2Dм
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpт
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_24/BiasAddд
sequential_4/conv2d_24/ReluRelu'sequential_4/conv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_24/ReluЫ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool)sequential_4/conv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_24/MaxPool╗
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_4/flatten_4/Constл
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2 
sequential_4/flatten_4/ReshapeЛ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluГ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluГ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/SoftmaxТ
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulП
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulП
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul┘	
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_20/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
х
e
F__inference_dropout_14_layer_call_and_return_conditional_losses_589316

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
__inference_call_532566

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource: D
6sequential_4_conv2d_20_biasadd_readvariableop_resource: P
5sequential_4_conv2d_21_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_21_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_22_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_22_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_23_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_23_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_24_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_12_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_20/BiasAdd/ReadVariableOpб,sequential_4/conv2d_20/Conv2D/ReadVariableOpб-sequential_4/conv2d_21/BiasAdd/ReadVariableOpб,sequential_4/conv2d_21/Conv2D/ReadVariableOpб-sequential_4/conv2d_22/BiasAdd/ReadVariableOpб,sequential_4/conv2d_22/Conv2D/ReadVariableOpб-sequential_4/conv2d_23/BiasAdd/ReadVariableOpб,sequential_4/conv2d_23/Conv2D/ReadVariableOpб-sequential_4/conv2d_24/BiasAdd/ReadVariableOpб,sequential_4/conv2d_24/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2с
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ђKK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1║
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:ђKK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_20/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_20/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK *
paddingSAME*
strides
2
sequential_4/conv2d_20/Conv2DЛ
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp▄
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK 2 
sequential_4/conv2d_20/BiasAddЮ
sequential_4/conv2d_20/ReluRelu'sequential_4/conv2d_20/BiasAdd:output:0*
T0*'
_output_shapes
:ђKK 2
sequential_4/conv2d_20/Reluж
%sequential_4/max_pooling2d_20/MaxPoolMaxPool)sequential_4/conv2d_20/Relu:activations:0*'
_output_shapes
:ђ%% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_20/MaxPool█
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_21/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_21/Conv2DConv2D.sequential_4/max_pooling2d_20/MaxPool:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_21/Conv2Dм
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpП
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ2 
sequential_4/conv2d_21/BiasAddъ
sequential_4/conv2d_21/ReluRelu'sequential_4/conv2d_21/BiasAdd:output:0*
T0*(
_output_shapes
:ђ%%ђ2
sequential_4/conv2d_21/ReluЖ
%sequential_4/max_pooling2d_21/MaxPoolMaxPool)sequential_4/conv2d_21/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_21/MaxPool▄
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_22/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_22/Conv2DConv2D.sequential_4/max_pooling2d_21/MaxPool:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_4/conv2d_22/Conv2Dм
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpП
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_4/conv2d_22/BiasAddъ
sequential_4/conv2d_22/ReluRelu'sequential_4/conv2d_22/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_4/conv2d_22/ReluЖ
%sequential_4/max_pooling2d_22/MaxPoolMaxPool)sequential_4/conv2d_22/Relu:activations:0*(
_output_shapes
:ђ		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_22/MaxPool▄
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_23/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_23/Conv2DConv2D.sequential_4/max_pooling2d_22/MaxPool:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ		ђ*
paddingSAME*
strides
2
sequential_4/conv2d_23/Conv2Dм
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpП
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ		ђ2 
sequential_4/conv2d_23/BiasAddъ
sequential_4/conv2d_23/ReluRelu'sequential_4/conv2d_23/BiasAdd:output:0*
T0*(
_output_shapes
:ђ		ђ2
sequential_4/conv2d_23/ReluЖ
%sequential_4/max_pooling2d_23/MaxPoolMaxPool)sequential_4/conv2d_23/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_23/MaxPool▄
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_24/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_24/Conv2DConv2D.sequential_4/max_pooling2d_23/MaxPool:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_4/conv2d_24/Conv2Dм
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpП
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_4/conv2d_24/BiasAddъ
sequential_4/conv2d_24/ReluRelu'sequential_4/conv2d_24/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_4/conv2d_24/ReluЖ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool)sequential_4/conv2d_24/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_24/MaxPool│
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*(
_output_shapes
:ђђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_4/flatten_4/Const╚
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0* 
_output_shapes
:
ђђ2 
sequential_4/flatten_4/ReshapeЛ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp¤
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOpм
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/BiasAddЊ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/ReluЦ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOpЛ
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOpм
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/BiasAddЊ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/ReluЦ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOpЕ
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЮ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_14/BiasAddt
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*
_output_shapes
:	ђ2
dense_14/Softmax┤
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*
_output_shapes
:	ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ђKK: : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:ђKK
 
_user_specified_nameinputs
▒
і
-__inference_sequential_4_layer_call_fn_588765
lambda_4_input
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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ
identityѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_5864932
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
_user_specified_namelambda_4_input
я
│
%__inference_CNN3_layer_call_fn_588223

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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

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
@__inference_CNN3_layer_call_and_return_conditional_losses_5873352
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
ЂЉ
С
__inference_call_532478

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource: D
6sequential_4_conv2d_20_biasadd_readvariableop_resource: P
5sequential_4_conv2d_21_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_21_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_22_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_22_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_23_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_23_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_24_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_12_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_20/BiasAdd/ReadVariableOpб,sequential_4/conv2d_20/Conv2D/ReadVariableOpб-sequential_4/conv2d_21/BiasAdd/ReadVariableOpб,sequential_4/conv2d_21/Conv2D/ReadVariableOpб-sequential_4/conv2d_22/BiasAdd/ReadVariableOpб,sequential_4/conv2d_22/Conv2D/ReadVariableOpб-sequential_4/conv2d_23/BiasAdd/ReadVariableOpб,sequential_4/conv2d_23/Conv2D/ReadVariableOpб-sequential_4/conv2d_24/BiasAdd/ReadVariableOpб,sequential_4/conv2d_24/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2с
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ђKK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1║
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:ђKK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_20/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_20/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK *
paddingSAME*
strides
2
sequential_4/conv2d_20/Conv2DЛ
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp▄
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK 2 
sequential_4/conv2d_20/BiasAddЮ
sequential_4/conv2d_20/ReluRelu'sequential_4/conv2d_20/BiasAdd:output:0*
T0*'
_output_shapes
:ђKK 2
sequential_4/conv2d_20/Reluж
%sequential_4/max_pooling2d_20/MaxPoolMaxPool)sequential_4/conv2d_20/Relu:activations:0*'
_output_shapes
:ђ%% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_20/MaxPool█
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_21/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_21/Conv2DConv2D.sequential_4/max_pooling2d_20/MaxPool:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_21/Conv2Dм
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpП
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ2 
sequential_4/conv2d_21/BiasAddъ
sequential_4/conv2d_21/ReluRelu'sequential_4/conv2d_21/BiasAdd:output:0*
T0*(
_output_shapes
:ђ%%ђ2
sequential_4/conv2d_21/ReluЖ
%sequential_4/max_pooling2d_21/MaxPoolMaxPool)sequential_4/conv2d_21/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_21/MaxPool▄
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_22/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_22/Conv2DConv2D.sequential_4/max_pooling2d_21/MaxPool:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_4/conv2d_22/Conv2Dм
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpП
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_4/conv2d_22/BiasAddъ
sequential_4/conv2d_22/ReluRelu'sequential_4/conv2d_22/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_4/conv2d_22/ReluЖ
%sequential_4/max_pooling2d_22/MaxPoolMaxPool)sequential_4/conv2d_22/Relu:activations:0*(
_output_shapes
:ђ		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_22/MaxPool▄
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_23/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_23/Conv2DConv2D.sequential_4/max_pooling2d_22/MaxPool:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ		ђ*
paddingSAME*
strides
2
sequential_4/conv2d_23/Conv2Dм
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpП
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ		ђ2 
sequential_4/conv2d_23/BiasAddъ
sequential_4/conv2d_23/ReluRelu'sequential_4/conv2d_23/BiasAdd:output:0*
T0*(
_output_shapes
:ђ		ђ2
sequential_4/conv2d_23/ReluЖ
%sequential_4/max_pooling2d_23/MaxPoolMaxPool)sequential_4/conv2d_23/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_23/MaxPool▄
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_24/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_24/Conv2DConv2D.sequential_4/max_pooling2d_23/MaxPool:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_4/conv2d_24/Conv2Dм
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpП
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_4/conv2d_24/BiasAddъ
sequential_4/conv2d_24/ReluRelu'sequential_4/conv2d_24/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_4/conv2d_24/ReluЖ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool)sequential_4/conv2d_24/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_24/MaxPool│
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*(
_output_shapes
:ђђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_4/flatten_4/Const╚
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0* 
_output_shapes
:
ђђ2 
sequential_4/flatten_4/ReshapeЛ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp¤
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOpм
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/BiasAddЊ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/ReluЦ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOpЛ
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOpм
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/BiasAddЊ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/ReluЦ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOpЕ
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЮ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_14/BiasAddt
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*
_output_shapes
:	ђ2
dense_14/Softmax┤
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*
_output_shapes
:	ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ђKK: : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:ђKK
 
_user_specified_nameinputs
Ў
ѓ
-__inference_sequential_4_layer_call_fn_588806

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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

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
H__inference_sequential_4_layer_call_and_return_conditional_losses_5864932
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
Ђ4
н
@__inference_CNN3_layer_call_and_return_conditional_losses_587171

inputs!
sequential_4_587098:!
sequential_4_587100:!
sequential_4_587102:!
sequential_4_587104:-
sequential_4_587106: !
sequential_4_587108: .
sequential_4_587110: ђ"
sequential_4_587112:	ђ/
sequential_4_587114:ђђ"
sequential_4_587116:	ђ/
sequential_4_587118:ђђ"
sequential_4_587120:	ђ/
sequential_4_587122:ђђ"
sequential_4_587124:	ђ'
sequential_4_587126:
ђђ"
sequential_4_587128:	ђ'
sequential_4_587130:
ђђ"
sequential_4_587132:	ђ"
dense_14_587147:	ђ
dense_14_587149:
identityѕб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpб dense_14/StatefulPartitionedCallб$sequential_4/StatefulPartitionedCallъ
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_587098sequential_4_587100sequential_4_587102sequential_4_587104sequential_4_587106sequential_4_587108sequential_4_587110sequential_4_587112sequential_4_587114sequential_4_587116sequential_4_587118sequential_4_587120sequential_4_587122sequential_4_587124sequential_4_587126sequential_4_587128sequential_4_587130sequential_4_587132*
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_5864932&
$sequential_4/StatefulPartitionedCall└
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0dense_14_587147dense_14_587149*
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
D__inference_dense_14_layer_call_and_return_conditional_losses_5871462"
 dense_14/StatefulPartitionedCall─
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_587106*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul╝
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_587126* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul╝
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_587130* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulС
IdentityIdentity)dense_14/StatefulPartitionedCall:output:03^conv2d_20/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
э
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_586710

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
┬
`
D__inference_lambda_4_layer_call_and_return_conditional_losses_588916

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
»
і
-__inference_sequential_4_layer_call_fn_588888
lambda_4_input
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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

unknown_14:	ђ

unknown_15:
ђђ

unknown_16:	ђ
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_5868592
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
_user_specified_namelambda_4_input
╣
г
D__inference_dense_13_layer_call_and_return_conditional_losses_589290

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЈ
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
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┬
`
D__inference_lambda_4_layer_call_and_return_conditional_losses_586737

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
@__inference_CNN3_layer_call_and_return_conditional_losses_587728

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource: D
6sequential_4_conv2d_20_biasadd_readvariableop_resource: P
5sequential_4_conv2d_21_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_21_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_22_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_22_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_23_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_23_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_24_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_12_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_20/BiasAdd/ReadVariableOpб,sequential_4/conv2d_20/Conv2D/ReadVariableOpб-sequential_4/conv2d_21/BiasAdd/ReadVariableOpб,sequential_4/conv2d_21/Conv2D/ReadVariableOpб-sequential_4/conv2d_22/BiasAdd/ReadVariableOpб,sequential_4/conv2d_22/Conv2D/ReadVariableOpб-sequential_4/conv2d_23/BiasAdd/ReadVariableOpб,sequential_4/conv2d_23/Conv2D/ReadVariableOpб-sequential_4/conv2d_24/BiasAdd/ReadVariableOpб,sequential_4/conv2d_24/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2в
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_20/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_20/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_20/Conv2DЛ
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpС
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_20/BiasAddЦ
sequential_4/conv2d_20/ReluRelu'sequential_4/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_20/Reluы
%sequential_4/max_pooling2d_20/MaxPoolMaxPool)sequential_4/conv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_20/MaxPool█
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_21/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_21/Conv2DConv2D.sequential_4/max_pooling2d_20/MaxPool:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_21/Conv2Dм
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpт
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_21/BiasAddд
sequential_4/conv2d_21/ReluRelu'sequential_4/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_21/ReluЫ
%sequential_4/max_pooling2d_21/MaxPoolMaxPool)sequential_4/conv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_21/MaxPool▄
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_22/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_22/Conv2DConv2D.sequential_4/max_pooling2d_21/MaxPool:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_22/Conv2Dм
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpт
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_22/BiasAddд
sequential_4/conv2d_22/ReluRelu'sequential_4/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_22/ReluЫ
%sequential_4/max_pooling2d_22/MaxPoolMaxPool)sequential_4/conv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_22/MaxPool▄
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_23/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_23/Conv2DConv2D.sequential_4/max_pooling2d_22/MaxPool:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_4/conv2d_23/Conv2Dм
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpт
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_4/conv2d_23/BiasAddд
sequential_4/conv2d_23/ReluRelu'sequential_4/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_4/conv2d_23/ReluЫ
%sequential_4/max_pooling2d_23/MaxPoolMaxPool)sequential_4/conv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_23/MaxPool▄
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_24/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_24/Conv2DConv2D.sequential_4/max_pooling2d_23/MaxPool:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_24/Conv2Dм
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpт
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_24/BiasAddд
sequential_4/conv2d_24/ReluRelu'sequential_4/conv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_24/ReluЫ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool)sequential_4/conv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_24/MaxPool╗
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_4/flatten_4/Constл
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2 
sequential_4/flatten_4/ReshapeЛ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluГ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluГ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/SoftmaxТ
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulП
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulП
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul┘	
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_20/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┬
`
D__inference_lambda_4_layer_call_and_return_conditional_losses_588924

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
┼▀
Ћ
@__inference_CNN3_layer_call_and_return_conditional_losses_588088
input_1H
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource: D
6sequential_4_conv2d_20_biasadd_readvariableop_resource: P
5sequential_4_conv2d_21_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_21_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_22_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_22_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_23_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_23_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_24_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_12_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpб1sequential_4/batch_normalization_4/AssignNewValueб3sequential_4/batch_normalization_4/AssignNewValue_1бBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_20/BiasAdd/ReadVariableOpб,sequential_4/conv2d_20/Conv2D/ReadVariableOpб-sequential_4/conv2d_21/BiasAdd/ReadVariableOpб,sequential_4/conv2d_21/Conv2D/ReadVariableOpб-sequential_4/conv2d_22/BiasAdd/ReadVariableOpб,sequential_4/conv2d_22/Conv2D/ReadVariableOpб-sequential_4/conv2d_23/BiasAdd/ReadVariableOpб,sequential_4/conv2d_23/Conv2D/ReadVariableOpб-sequential_4/conv2d_24/BiasAdd/ReadVariableOpб,sequential_4/conv2d_24/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2В
#sequential_4/lambda_4/strided_sliceStridedSliceinput_12sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1л
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<25
3sequential_4/batch_normalization_4/FusedBatchNormV3ы
1sequential_4/batch_normalization_4/AssignNewValueAssignVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource@sequential_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_4/batch_normalization_4/AssignNewValue§
3sequential_4/batch_normalization_4/AssignNewValue_1AssignVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceDsequential_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0E^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_4/batch_normalization_4/AssignNewValue_1┌
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_20/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_20/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_20/Conv2DЛ
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpС
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_20/BiasAddЦ
sequential_4/conv2d_20/ReluRelu'sequential_4/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_20/Reluы
%sequential_4/max_pooling2d_20/MaxPoolMaxPool)sequential_4/conv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_20/MaxPool█
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_21/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_21/Conv2DConv2D.sequential_4/max_pooling2d_20/MaxPool:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_21/Conv2Dм
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpт
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_21/BiasAddд
sequential_4/conv2d_21/ReluRelu'sequential_4/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_21/ReluЫ
%sequential_4/max_pooling2d_21/MaxPoolMaxPool)sequential_4/conv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_21/MaxPool▄
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_22/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_22/Conv2DConv2D.sequential_4/max_pooling2d_21/MaxPool:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_22/Conv2Dм
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpт
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_22/BiasAddд
sequential_4/conv2d_22/ReluRelu'sequential_4/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_22/ReluЫ
%sequential_4/max_pooling2d_22/MaxPoolMaxPool)sequential_4/conv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_22/MaxPool▄
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_23/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_23/Conv2DConv2D.sequential_4/max_pooling2d_22/MaxPool:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
sequential_4/conv2d_23/Conv2Dм
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpт
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2 
sequential_4/conv2d_23/BiasAddд
sequential_4/conv2d_23/ReluRelu'sequential_4/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
sequential_4/conv2d_23/ReluЫ
%sequential_4/max_pooling2d_23/MaxPoolMaxPool)sequential_4/conv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_23/MaxPool▄
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_24/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_24/Conv2DConv2D.sequential_4/max_pooling2d_23/MaxPool:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_24/Conv2Dм
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpт
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_24/BiasAddд
sequential_4/conv2d_24/ReluRelu'sequential_4/conv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_24/ReluЫ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool)sequential_4/conv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_24/MaxPoolЊ
%sequential_4/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2'
%sequential_4/dropout_12/dropout/ConstВ
#sequential_4/dropout_12/dropout/MulMul.sequential_4/max_pooling2d_24/MaxPool:output:0.sequential_4/dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2%
#sequential_4/dropout_12/dropout/Mulг
%sequential_4/dropout_12/dropout/ShapeShape.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_12/dropout/ShapeЁ
<sequential_4/dropout_12/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_12/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_4/dropout_12/dropout/GreaterEqual/yД
,sequential_4/dropout_12/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_12/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2.
,sequential_4/dropout_12/dropout/GreaterEqualл
$sequential_4/dropout_12/dropout/CastCast0sequential_4/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2&
$sequential_4/dropout_12/dropout/Castс
%sequential_4/dropout_12/dropout/Mul_1Mul'sequential_4/dropout_12/dropout/Mul:z:0(sequential_4/dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2'
%sequential_4/dropout_12/dropout/Mul_1Ї
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential_4/flatten_4/Constл
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/dropout/Mul_1:z:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2 
sequential_4/flatten_4/ReshapeЛ
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluЊ
%sequential_4/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_4/dropout_13/dropout/Constя
#sequential_4/dropout_13/dropout/MulMul(sequential_4/dense_12/Relu:activations:0.sequential_4/dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_4/dropout_13/dropout/Mulд
%sequential_4/dropout_13/dropout/ShapeShape(sequential_4/dense_12/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_13/dropout/Shape§
<sequential_4/dropout_13/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_13/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_4/dropout_13/dropout/GreaterEqual/yЪ
,sequential_4/dropout_13/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_13/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_4/dropout_13/dropout/GreaterEqual╚
$sequential_4/dropout_13/dropout/CastCast0sequential_4/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_4/dropout_13/dropout/Cast█
%sequential_4/dropout_13/dropout/Mul_1Mul'sequential_4/dropout_13/dropout/Mul:z:0(sequential_4/dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_4/dropout_13/dropout/Mul_1Л
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/dropout/Mul_1:z:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluЊ
%sequential_4/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_4/dropout_14/dropout/Constя
#sequential_4/dropout_14/dropout/MulMul(sequential_4/dense_13/Relu:activations:0.sequential_4/dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_4/dropout_14/dropout/Mulд
%sequential_4/dropout_14/dropout/ShapeShape(sequential_4/dense_13/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_14/dropout/Shape§
<sequential_4/dropout_14/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_14/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_4/dropout_14/dropout/GreaterEqual/yЪ
,sequential_4/dropout_14/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_14/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_4/dropout_14/dropout/GreaterEqual╚
$sequential_4/dropout_14/dropout/CastCast0sequential_4/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_4/dropout_14/dropout/Cast█
%sequential_4/dropout_14/dropout/Mul_1Mul'sequential_4/dropout_14/dropout/Mul:z:0(sequential_4/dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_4/dropout_14/dropout/Mul_1Е
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/SoftmaxТ
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulП
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulП
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul├

IdentityIdentitydense_14/Softmax:softmax:03^conv2d_20/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^sequential_4/batch_normalization_4/AssignNewValue4^sequential_4/batch_normalization_4/AssignNewValue_1C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1sequential_4/batch_normalization_4/AssignNewValue1sequential_4/batch_normalization_4/AssignNewValue2j
3sequential_4/batch_normalization_4/AssignNewValue_13sequential_4/batch_normalization_4/AssignNewValue_12ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
А
Ђ
E__inference_conv2d_22_layer_call_and_return_conditional_losses_586356

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
О
F
*__inference_flatten_4_layer_call_fn_589208

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
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_5864122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
э
d
F__inference_dropout_14_layer_call_and_return_conditional_losses_589304

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
╔
G
+__inference_dropout_14_layer_call_fn_589321

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
F__inference_dropout_14_layer_call_and_return_conditional_losses_5864722
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
Л
б
*__inference_conv2d_22_layer_call_fn_589130

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
E__inference_conv2d_22_layer_call_and_return_conditional_losses_5863562
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
А
Ђ
E__inference_conv2d_23_layer_call_and_return_conditional_losses_589141

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
я
M
1__inference_max_pooling2d_23_layer_call_fn_586247

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
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_5862412
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
И

Ш
D__inference_dense_14_layer_call_and_return_conditional_losses_587146

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
├
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588988

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
Д
Ў
)__inference_dense_12_layer_call_fn_589240

inputs
unknown:
ђђ
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
D__inference_dense_12_layer_call_and_return_conditional_losses_5864312
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ж
G
+__inference_dropout_12_layer_call_fn_589192

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
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5864042
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ќ
ѓ
-__inference_sequential_4_layer_call_fn_588847

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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

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
H__inference_sequential_4_layer_call_and_return_conditional_losses_5868592
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
┐
│
E__inference_conv2d_20_layer_call_and_return_conditional_losses_586320

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpЋ
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
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
├
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_586293

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
И

Ш
D__inference_dense_14_layer_call_and_return_conditional_losses_588899

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
І
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_586095

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
Ю
ђ
E__inference_conv2d_21_layer_call_and_return_conditional_losses_589101

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
э
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_589006

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
Ш
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_589187

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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
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
:         ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
ш
d
+__inference_dropout_12_layer_call_fn_589197

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
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5866242
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Н
d
+__inference_dropout_14_layer_call_fn_589326

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
F__inference_dropout_14_layer_call_and_return_conditional_losses_5865522
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
х
e
F__inference_dropout_14_layer_call_and_return_conditional_losses_586552

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
ЌЋ
К)
"__inference__traced_restore_589782
file_prefix3
 assignvariableop_dense_14_kernel:	ђ.
 assignvariableop_1_dense_14_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_4_gamma:;
-assignvariableop_8_batch_normalization_4_beta:=
#assignvariableop_9_conv2d_20_kernel: 0
"assignvariableop_10_conv2d_20_bias: ?
$assignvariableop_11_conv2d_21_kernel: ђ1
"assignvariableop_12_conv2d_21_bias:	ђ@
$assignvariableop_13_conv2d_22_kernel:ђђ1
"assignvariableop_14_conv2d_22_bias:	ђ@
$assignvariableop_15_conv2d_23_kernel:ђђ1
"assignvariableop_16_conv2d_23_bias:	ђ@
$assignvariableop_17_conv2d_24_kernel:ђђ1
"assignvariableop_18_conv2d_24_bias:	ђ7
#assignvariableop_19_dense_12_kernel:
ђђ0
!assignvariableop_20_dense_12_bias:	ђ7
#assignvariableop_21_dense_13_kernel:
ђђ0
!assignvariableop_22_dense_13_bias:	ђC
5assignvariableop_23_batch_normalization_4_moving_mean:G
9assignvariableop_24_batch_normalization_4_moving_variance:#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: =
*assignvariableop_29_adam_dense_14_kernel_m:	ђ6
(assignvariableop_30_adam_dense_14_bias_m:D
6assignvariableop_31_adam_batch_normalization_4_gamma_m:C
5assignvariableop_32_adam_batch_normalization_4_beta_m:E
+assignvariableop_33_adam_conv2d_20_kernel_m: 7
)assignvariableop_34_adam_conv2d_20_bias_m: F
+assignvariableop_35_adam_conv2d_21_kernel_m: ђ8
)assignvariableop_36_adam_conv2d_21_bias_m:	ђG
+assignvariableop_37_adam_conv2d_22_kernel_m:ђђ8
)assignvariableop_38_adam_conv2d_22_bias_m:	ђG
+assignvariableop_39_adam_conv2d_23_kernel_m:ђђ8
)assignvariableop_40_adam_conv2d_23_bias_m:	ђG
+assignvariableop_41_adam_conv2d_24_kernel_m:ђђ8
)assignvariableop_42_adam_conv2d_24_bias_m:	ђ>
*assignvariableop_43_adam_dense_12_kernel_m:
ђђ7
(assignvariableop_44_adam_dense_12_bias_m:	ђ>
*assignvariableop_45_adam_dense_13_kernel_m:
ђђ7
(assignvariableop_46_adam_dense_13_bias_m:	ђ=
*assignvariableop_47_adam_dense_14_kernel_v:	ђ6
(assignvariableop_48_adam_dense_14_bias_v:D
6assignvariableop_49_adam_batch_normalization_4_gamma_v:C
5assignvariableop_50_adam_batch_normalization_4_beta_v:E
+assignvariableop_51_adam_conv2d_20_kernel_v: 7
)assignvariableop_52_adam_conv2d_20_bias_v: F
+assignvariableop_53_adam_conv2d_21_kernel_v: ђ8
)assignvariableop_54_adam_conv2d_21_bias_v:	ђG
+assignvariableop_55_adam_conv2d_22_kernel_v:ђђ8
)assignvariableop_56_adam_conv2d_22_bias_v:	ђG
+assignvariableop_57_adam_conv2d_23_kernel_v:ђђ8
)assignvariableop_58_adam_conv2d_23_bias_v:	ђG
+assignvariableop_59_adam_conv2d_24_kernel_v:ђђ8
)assignvariableop_60_adam_conv2d_24_bias_v:	ђ>
*assignvariableop_61_adam_dense_12_kernel_v:
ђђ7
(assignvariableop_62_adam_dense_12_bias_v:	ђ>
*assignvariableop_63_adam_dense_13_kernel_v:
ђђ7
(assignvariableop_64_adam_dense_13_bias_v:	ђ
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
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_4_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▓
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_4_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9е
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_20_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ф
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_20_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_21_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ф
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_21_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_22_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ф
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_22_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15г
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_23_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ф
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_23_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17г
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_24_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ф
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_24_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ф
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_12_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Е
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_12_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ф
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_13_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Е
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_13_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23й
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_4_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24┴
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_4_moving_varianceIdentity_24:output:0"/device:CPU:0*
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
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_14_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30░
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_14_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Й
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_4_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32й
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_batch_normalization_4_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_20_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_20_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35│
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_21_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▒
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_21_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37│
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_22_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_22_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39│
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_23_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40▒
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_23_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41│
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_24_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42▒
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_24_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▓
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_12_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44░
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_12_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45▓
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_13_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46░
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_13_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47▓
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_14_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48░
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_14_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Й
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_4_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50й
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_4_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51│
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_20_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52▒
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_20_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53│
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_21_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54▒
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_21_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55│
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_22_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56▒
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_22_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57│
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_23_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58▒
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_23_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59│
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_24_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60▒
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_24_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61▓
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_12_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62░
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_12_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63▓
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_13_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64░
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_13_bias_vIdentity_64:output:0"/device:CPU:0*
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
э
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_586442

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
В
Л
6__inference_batch_normalization_4_layer_call_fn_589032

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5861392
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
┐
│
E__inference_conv2d_20_layer_call_and_return_conditional_losses_589081

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpЋ
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
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
х
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_586585

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
А
Ђ
E__inference_conv2d_24_layer_call_and_return_conditional_losses_589161

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
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
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

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
╣
г
D__inference_dense_13_layer_call_and_return_conditional_losses_586461

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЈ
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
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_586241

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
╔
G
+__inference_dropout_13_layer_call_fn_589262

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
F__inference_dropout_13_layer_call_and_return_conditional_losses_5864422
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
Фђ
┐
__inference__traced_save_589577
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop6
2savev2_adam_conv2d_21_kernel_m_read_readvariableop4
0savev2_adam_conv2d_21_bias_m_read_readvariableop6
2savev2_adam_conv2d_22_kernel_m_read_readvariableop4
0savev2_adam_conv2d_22_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop6
2savev2_adam_conv2d_21_kernel_v_read_readvariableop4
0savev2_adam_conv2d_21_bias_v_read_readvariableop6
2savev2_adam_conv2d_22_kernel_v_read_readvariableop4
0savev2_adam_conv2d_22_bias_v_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop2savev2_adam_conv2d_21_kernel_m_read_readvariableop0savev2_adam_conv2d_21_bias_m_read_readvariableop2savev2_adam_conv2d_22_kernel_m_read_readvariableop0savev2_adam_conv2d_22_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop2savev2_adam_conv2d_21_kernel_v_read_readvariableop0savev2_adam_conv2d_21_bias_v_read_readvariableop2savev2_adam_conv2d_22_kernel_v_read_readvariableop0savev2_adam_conv2d_22_bias_v_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
З: :	ђ:: : : : : ::: : : ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:
ђђ:ђ:
ђђ:ђ::: : : : :	ђ:::: : : ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:
ђђ:ђ:
ђђ:ђ:	ђ:::: : : ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:
ђђ:ђ:
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
:ђђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!
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
:ђђ:!+

_output_shapes	
:ђ:&,"
 
_output_shapes
:
ђђ:!-
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
:ђђ:!=

_output_shapes	
:ђ:&>"
 
_output_shapes
:
ђђ:!?
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
кЇ
Д
H__inference_sequential_4_layer_call_and_return_conditional_losses_588385

inputs;
-batch_normalization_4_readvariableop_resource:=
/batch_normalization_4_readvariableop_1_resource:L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_20_conv2d_readvariableop_resource: 7
)conv2d_20_biasadd_readvariableop_resource: C
(conv2d_21_conv2d_readvariableop_resource: ђ8
)conv2d_21_biasadd_readvariableop_resource:	ђD
(conv2d_22_conv2d_readvariableop_resource:ђђ8
)conv2d_22_biasadd_readvariableop_resource:	ђD
(conv2d_23_conv2d_readvariableop_resource:ђђ8
)conv2d_23_biasadd_readvariableop_resource:	ђD
(conv2d_24_conv2d_readvariableop_resource:ђђ8
)conv2d_24_biasadd_readvariableop_resource:	ђ;
'dense_12_matmul_readvariableop_resource:
ђђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identityѕб5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б conv2d_20/BiasAdd/ReadVariableOpбconv2d_20/Conv2D/ReadVariableOpб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб conv2d_21/BiasAdd/ReadVariableOpбconv2d_21/Conv2D/ReadVariableOpб conv2d_22/BiasAdd/ReadVariableOpбconv2d_22/Conv2D/ReadVariableOpб conv2d_23/BiasAdd/ReadVariableOpбconv2d_23/Conv2D/ReadVariableOpб conv2d_24/BiasAdd/ReadVariableOpбconv2d_24/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stackЎ
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1Ў
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2ф
lambda_4/strided_sliceStridedSliceinputs%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_4/strided_sliceХ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1у
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3│
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_20/Conv2D/ReadVariableOpт
conv2d_20/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_20/Conv2Dф
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp░
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_20/Relu╩
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool┤
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_21/Conv2D/ReadVariableOpП
conv2d_21/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_21/Conv2DФ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp▒
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_21/BiasAdd
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_21/Relu╦
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPoolх
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_22/Conv2D/ReadVariableOpП
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_22/Conv2DФ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp▒
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_22/Relu╦
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPoolх
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_23/Conv2D/ReadVariableOpП
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ*
paddingSAME*
strides
2
conv2d_23/Conv2DФ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp▒
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		ђ2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:         		ђ2
conv2d_23/Relu╦
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPoolх
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_24/Conv2D/ReadVariableOpП
conv2d_24/Conv2DConv2D!max_pooling2d_23/MaxPool:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_24/Conv2DФ
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp▒
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_24/BiasAdd
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_24/Relu╦
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_24/MaxPoolћ
dropout_12/IdentityIdentity!max_pooling2d_24/MaxPool:output:0*
T0*0
_output_shapes
:         ђ2
dropout_12/Identitys
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_4/Constю
flatten_4/ReshapeReshapedropout_12/Identity:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_4/Reshapeф
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/MatMulе
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_12/BiasAdd/ReadVariableOpд
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_12/Reluє
dropout_13/IdentityIdentitydense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_13/Identityф
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_13/MatMul/ReadVariableOpЦ
dense_13/MatMulMatMuldropout_13/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/MatMulе
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_13/BiasAdd/ReadVariableOpд
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_13/Reluє
dropout_14/IdentityIdentitydense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_14/Identity┘
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mulл
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulл
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul»
IdentityIdentitydropout_14/Identity:output:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_586205

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
А
Ђ
E__inference_conv2d_22_layer_call_and_return_conditional_losses_589121

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
я
M
1__inference_max_pooling2d_24_layer_call_fn_586259

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
L__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_5862532
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
╬
А
*__inference_conv2d_21_layer_call_fn_589110

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
E__inference_conv2d_21_layer_call_and_return_conditional_losses_5863382
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
Ќ
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_589175

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_586253

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
 3
н
@__inference_CNN3_layer_call_and_return_conditional_losses_587335

inputs!
sequential_4_587274:!
sequential_4_587276:!
sequential_4_587278:!
sequential_4_587280:-
sequential_4_587282: !
sequential_4_587284: .
sequential_4_587286: ђ"
sequential_4_587288:	ђ/
sequential_4_587290:ђђ"
sequential_4_587292:	ђ/
sequential_4_587294:ђђ"
sequential_4_587296:	ђ/
sequential_4_587298:ђђ"
sequential_4_587300:	ђ'
sequential_4_587302:
ђђ"
sequential_4_587304:	ђ'
sequential_4_587306:
ђђ"
sequential_4_587308:	ђ"
dense_14_587311:	ђ
dense_14_587313:
identityѕб2conv2d_20/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpб dense_14/StatefulPartitionedCallб$sequential_4/StatefulPartitionedCallю
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_587274sequential_4_587276sequential_4_587278sequential_4_587280sequential_4_587282sequential_4_587284sequential_4_587286sequential_4_587288sequential_4_587290sequential_4_587292sequential_4_587294sequential_4_587296sequential_4_587298sequential_4_587300sequential_4_587302sequential_4_587304sequential_4_587306sequential_4_587308*
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_5868592&
$sequential_4/StatefulPartitionedCall└
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0dense_14_587311dense_14_587313*
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
D__inference_dense_14_layer_call_and_return_conditional_losses_5871462"
 dense_14/StatefulPartitionedCall─
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_587282*&
_output_shapes
: *
dtype024
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_20/kernel/Regularizer/SquareА
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_20/kernel/Regularizer/Const┬
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/SumЇ
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_20/kernel/Regularizer/mul/x─
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_20/kernel/Regularizer/mul╝
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_587302* 
_output_shapes
:
ђђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOpИ
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul╝
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_587306* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulС
IdentityIdentity)dense_14/StatefulPartitionedCall:output:03^conv2d_20/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         KK: : : : : : : : : : : : : : : : : : : : 2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
э
d
F__inference_dropout_14_layer_call_and_return_conditional_losses_586472

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
с
┤
%__inference_CNN3_layer_call_fn_588133
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

unknown_11:ђђ

unknown_12:	ђ

unknown_13:
ђђ

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
@__inference_CNN3_layer_call_and_return_conditional_losses_5871712
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
StatefulPartitionedCall:0         tensorflow/serving/predict:ГЧ
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
+Ю&call_and_return_all_conditional_losses
ъ__call__
Ъ_default_save_signature
	аcall"Ј	
_tf_keras_modelш{"name": "CNN3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN3", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN3"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
оѕ
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
+А&call_and_return_all_conditional_losses
б__call__"ѓё
_tf_keras_sequentialРЃ{"name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_4_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 42, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_4_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_4_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 29}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 30}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 31}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 34}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 41}]}}}
О

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+Б&call_and_return_all_conditional_losses
ц__call__"░
_tf_keras_layerќ{"name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
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
regularization_losses
<layer_metrics
trainable_variables

=layers
>metrics
?non_trainable_variables
@layer_regularization_losses
	variables
ъ__call__
Ъ_default_save_signature
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
-
Цserving_default"
signature_map
п
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+д&call_and_return_all_conditional_losses
Д__call__"К
_tf_keras_layerГ{"name": "lambda_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
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
+е&call_and_return_all_conditional_losses
Е__call__"Ь
_tf_keras_layerн{"name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
б

,kernel
-bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+ф&call_and_return_all_conditional_losses
Ф__call__"ч	
_tf_keras_layerр	{"name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
│
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+г&call_and_return_all_conditional_losses
Г__call__"б
_tf_keras_layerѕ{"name": "max_pooling2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 49}}
о


.kernel
/bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+«&call_and_return_all_conditional_losses
»__call__"»	
_tf_keras_layerЋ	{"name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
│
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"б
_tf_keras_layerѕ{"name": "max_pooling2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 51}}
п


0kernel
1bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"▒	
_tf_keras_layerЌ	{"name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
│
^regularization_losses
_trainable_variables
`	variables
a	keras_api
+┤&call_and_return_all_conditional_losses
х__call__"б
_tf_keras_layerѕ{"name": "max_pooling2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 53}}
о


2kernel
3bias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
+Х&call_and_return_all_conditional_losses
и__call__"»	
_tf_keras_layerЋ	{"name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 9, 9, 256]}}
│
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
+И&call_and_return_all_conditional_losses
╣__call__"б
_tf_keras_layerѕ{"name": "max_pooling2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 55}}
о


4kernel
5bias
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
+║&call_and_return_all_conditional_losses
╗__call__"»	
_tf_keras_layerЋ	{"name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 4, 4, 512]}}
│
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+╝&call_and_return_all_conditional_losses
й__call__"б
_tf_keras_layerѕ{"name": "max_pooling2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 57}}
Ђ
rregularization_losses
strainable_variables
t	variables
u	keras_api
+Й&call_and_return_all_conditional_losses
┐__call__"­
_tf_keras_layerо{"name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 30}
ў
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
+└&call_and_return_all_conditional_losses
┴__call__"Є
_tf_keras_layerь{"name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 58}}
е	

6kernel
7bias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
+┬&call_and_return_all_conditional_losses
├__call__"Ђ
_tf_keras_layerу{"name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 34}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 2048]}}
Ѓ
~regularization_losses
trainable_variables
ђ	variables
Ђ	keras_api
+─&call_and_return_all_conditional_losses
┼__call__"­
_tf_keras_layerо{"name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 36}
ф	

8kernel
9bias
ѓregularization_losses
Ѓtrainable_variables
ё	variables
Ё	keras_api
+к&call_and_return_all_conditional_losses
К__call__" 
_tf_keras_layerт{"name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Ё
єregularization_losses
Єtrainable_variables
ѕ	variables
Ѕ	keras_api
+╚&call_and_return_all_conditional_losses
╔__call__"­
_tf_keras_layerо{"name": "dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 41}
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
regularization_losses
іlayer_metrics
trainable_variables
Іlayers
їmetrics
Їnon_trainable_variables
 јlayer_regularization_losses
	variables
б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
": 	ђ2dense_14/kernel
:2dense_14/bias
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
!regularization_losses
Јlayers
"trainable_variables
љmetrics
Љnon_trainable_variables
#	variables
 њlayer_regularization_losses
Њlayer_metrics
ц__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
*:( 2conv2d_20/kernel
: 2conv2d_20/bias
+:) ђ2conv2d_21/kernel
:ђ2conv2d_21/bias
,:*ђђ2conv2d_22/kernel
:ђ2conv2d_22/bias
,:*ђђ2conv2d_23/kernel
:ђ2conv2d_23/bias
,:*ђђ2conv2d_24/kernel
:ђ2conv2d_24/bias
#:!
ђђ2dense_12/kernel
:ђ2dense_12/bias
#:!
ђђ2dense_13/kernel
:ђ2dense_13/bias
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
0
ћ0
Ћ1"
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
 "
trackable_list_wrapper
х
Aregularization_losses
ќlayers
Btrainable_variables
Ќmetrics
ўnon_trainable_variables
C	variables
 Ўlayer_regularization_losses
џlayer_metrics
Д__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
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
Fregularization_losses
Џlayers
Gtrainable_variables
юmetrics
Юnon_trainable_variables
H	variables
 ъlayer_regularization_losses
Ъlayer_metrics
Е__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
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
Jregularization_losses
аlayers
Ktrainable_variables
Аmetrics
бnon_trainable_variables
L	variables
 Бlayer_regularization_losses
цlayer_metrics
Ф__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Nregularization_losses
Цlayers
Otrainable_variables
дmetrics
Дnon_trainable_variables
P	variables
 еlayer_regularization_losses
Еlayer_metrics
Г__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
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
Rregularization_losses
фlayers
Strainable_variables
Фmetrics
гnon_trainable_variables
T	variables
 Гlayer_regularization_losses
«layer_metrics
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Vregularization_losses
»layers
Wtrainable_variables
░metrics
▒non_trainable_variables
X	variables
 ▓layer_regularization_losses
│layer_metrics
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
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
Zregularization_losses
┤layers
[trainable_variables
хmetrics
Хnon_trainable_variables
\	variables
 иlayer_regularization_losses
Иlayer_metrics
│__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
^regularization_losses
╣layers
_trainable_variables
║metrics
╗non_trainable_variables
`	variables
 ╝layer_regularization_losses
йlayer_metrics
х__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
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
bregularization_losses
Йlayers
ctrainable_variables
┐metrics
└non_trainable_variables
d	variables
 ┴layer_regularization_losses
┬layer_metrics
и__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
fregularization_losses
├layers
gtrainable_variables
─metrics
┼non_trainable_variables
h	variables
 кlayer_regularization_losses
Кlayer_metrics
╣__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
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
jregularization_losses
╚layers
ktrainable_variables
╔metrics
╩non_trainable_variables
l	variables
 ╦layer_regularization_losses
╠layer_metrics
╗__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
nregularization_losses
═layers
otrainable_variables
╬metrics
¤non_trainable_variables
p	variables
 лlayer_regularization_losses
Лlayer_metrics
й__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
rregularization_losses
мlayers
strainable_variables
Мmetrics
нnon_trainable_variables
t	variables
 Нlayer_regularization_losses
оlayer_metrics
┐__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
vregularization_losses
Оlayers
wtrainable_variables
пmetrics
┘non_trainable_variables
x	variables
 ┌layer_regularization_losses
█layer_metrics
┴__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
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
zregularization_losses
▄layers
{trainable_variables
Пmetrics
яnon_trainable_variables
|	variables
 ▀layer_regularization_losses
Яlayer_metrics
├__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Х
~regularization_losses
рlayers
trainable_variables
Рmetrics
сnon_trainable_variables
ђ	variables
 Сlayer_regularization_losses
тlayer_metrics
┼__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
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
ѓregularization_losses
Тlayers
Ѓtrainable_variables
уmetrics
Уnon_trainable_variables
ё	variables
 жlayer_regularization_losses
Жlayer_metrics
К__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
єregularization_losses
вlayers
Єtrainable_variables
Вmetrics
ьnon_trainable_variables
ѕ	variables
 Ьlayer_regularization_losses
№layer_metrics
╔__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
:0
;1"
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
╩0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
╦0"
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
╠0"
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
':%	ђ2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
.:,2"Adam/batch_normalization_4/gamma/m
-:+2!Adam/batch_normalization_4/beta/m
/:- 2Adam/conv2d_20/kernel/m
!: 2Adam/conv2d_20/bias/m
0:. ђ2Adam/conv2d_21/kernel/m
": ђ2Adam/conv2d_21/bias/m
1:/ђђ2Adam/conv2d_22/kernel/m
": ђ2Adam/conv2d_22/bias/m
1:/ђђ2Adam/conv2d_23/kernel/m
": ђ2Adam/conv2d_23/bias/m
1:/ђђ2Adam/conv2d_24/kernel/m
": ђ2Adam/conv2d_24/bias/m
(:&
ђђ2Adam/dense_12/kernel/m
!:ђ2Adam/dense_12/bias/m
(:&
ђђ2Adam/dense_13/kernel/m
!:ђ2Adam/dense_13/bias/m
':%	ђ2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
.:,2"Adam/batch_normalization_4/gamma/v
-:+2!Adam/batch_normalization_4/beta/v
/:- 2Adam/conv2d_20/kernel/v
!: 2Adam/conv2d_20/bias/v
0:. ђ2Adam/conv2d_21/kernel/v
": ђ2Adam/conv2d_21/bias/v
1:/ђђ2Adam/conv2d_22/kernel/v
": ђ2Adam/conv2d_22/bias/v
1:/ђђ2Adam/conv2d_23/kernel/v
": ђ2Adam/conv2d_23/bias/v
1:/ђђ2Adam/conv2d_24/kernel/v
": ђ2Adam/conv2d_24/bias/v
(:&
ђђ2Adam/dense_12/kernel/v
!:ђ2Adam/dense_12/bias/v
(:&
ђђ2Adam/dense_13/kernel/v
!:ђ2Adam/dense_13/bias/v
┬2┐
@__inference_CNN3_layer_call_and_return_conditional_losses_587728
@__inference_CNN3_layer_call_and_return_conditional_losses_587855
@__inference_CNN3_layer_call_and_return_conditional_losses_587961
@__inference_CNN3_layer_call_and_return_conditional_losses_588088┤
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
о2М
%__inference_CNN3_layer_call_fn_588133
%__inference_CNN3_layer_call_fn_588178
%__inference_CNN3_layer_call_fn_588223
%__inference_CNN3_layer_call_fn_588268┤
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
!__inference__wrapped_model_586073Й
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
__inference_call_532478
__inference_call_532566
__inference_call_532654│
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
Ь2в
H__inference_sequential_4_layer_call_and_return_conditional_losses_588385
H__inference_sequential_4_layer_call_and_return_conditional_losses_588505
H__inference_sequential_4_layer_call_and_return_conditional_losses_588604
H__inference_sequential_4_layer_call_and_return_conditional_losses_588724└
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
ѓ2 
-__inference_sequential_4_layer_call_fn_588765
-__inference_sequential_4_layer_call_fn_588806
-__inference_sequential_4_layer_call_fn_588847
-__inference_sequential_4_layer_call_fn_588888└
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
Ь2в
D__inference_dense_14_layer_call_and_return_conditional_losses_588899б
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
)__inference_dense_14_layer_call_fn_588908б
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
$__inference_signature_wrapper_587622input_1"ћ
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
м2¤
D__inference_lambda_4_layer_call_and_return_conditional_losses_588916
D__inference_lambda_4_layer_call_and_return_conditional_losses_588924└
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
ю2Ў
)__inference_lambda_4_layer_call_fn_588929
)__inference_lambda_4_layer_call_fn_588934└
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
є2Ѓ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588952
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588970
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588988
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_589006┤
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
џ2Ќ
6__inference_batch_normalization_4_layer_call_fn_589019
6__inference_batch_normalization_4_layer_call_fn_589032
6__inference_batch_normalization_4_layer_call_fn_589045
6__inference_batch_normalization_4_layer_call_fn_589058┤
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
№2В
E__inference_conv2d_20_layer_call_and_return_conditional_losses_589081б
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
н2Л
*__inference_conv2d_20_layer_call_fn_589090б
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
┤2▒
L__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_586205Я
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
Ў2ќ
1__inference_max_pooling2d_20_layer_call_fn_586211Я
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
№2В
E__inference_conv2d_21_layer_call_and_return_conditional_losses_589101б
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
н2Л
*__inference_conv2d_21_layer_call_fn_589110б
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
┤2▒
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_586217Я
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
Ў2ќ
1__inference_max_pooling2d_21_layer_call_fn_586223Я
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
№2В
E__inference_conv2d_22_layer_call_and_return_conditional_losses_589121б
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
н2Л
*__inference_conv2d_22_layer_call_fn_589130б
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
┤2▒
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_586229Я
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
Ў2ќ
1__inference_max_pooling2d_22_layer_call_fn_586235Я
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
№2В
E__inference_conv2d_23_layer_call_and_return_conditional_losses_589141б
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
н2Л
*__inference_conv2d_23_layer_call_fn_589150б
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
┤2▒
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_586241Я
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
Ў2ќ
1__inference_max_pooling2d_23_layer_call_fn_586247Я
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
№2В
E__inference_conv2d_24_layer_call_and_return_conditional_losses_589161б
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
н2Л
*__inference_conv2d_24_layer_call_fn_589170б
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
┤2▒
L__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_586253Я
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
Ў2ќ
1__inference_max_pooling2d_24_layer_call_fn_586259Я
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
╩2К
F__inference_dropout_12_layer_call_and_return_conditional_losses_589175
F__inference_dropout_12_layer_call_and_return_conditional_losses_589187┤
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
ћ2Љ
+__inference_dropout_12_layer_call_fn_589192
+__inference_dropout_12_layer_call_fn_589197┤
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
№2В
E__inference_flatten_4_layer_call_and_return_conditional_losses_589203б
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
н2Л
*__inference_flatten_4_layer_call_fn_589208б
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
D__inference_dense_12_layer_call_and_return_conditional_losses_589231б
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
)__inference_dense_12_layer_call_fn_589240б
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
╩2К
F__inference_dropout_13_layer_call_and_return_conditional_losses_589245
F__inference_dropout_13_layer_call_and_return_conditional_losses_589257┤
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
ћ2Љ
+__inference_dropout_13_layer_call_fn_589262
+__inference_dropout_13_layer_call_fn_589267┤
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
Ь2в
D__inference_dense_13_layer_call_and_return_conditional_losses_589290б
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
)__inference_dense_13_layer_call_fn_589299б
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
╩2К
F__inference_dropout_14_layer_call_and_return_conditional_losses_589304
F__inference_dropout_14_layer_call_and_return_conditional_losses_589316┤
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
ћ2Љ
+__inference_dropout_14_layer_call_fn_589321
+__inference_dropout_14_layer_call_fn_589326┤
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
__inference_loss_fn_0_589337Ј
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
__inference_loss_fn_1_589348Ј
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
__inference_loss_fn_2_589359Ј
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
@__inference_CNN3_layer_call_and_return_conditional_losses_587728z*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p 
ф "%б"
і
0         
џ Й
@__inference_CNN3_layer_call_and_return_conditional_losses_587855z*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p
ф "%б"
і
0         
џ ┐
@__inference_CNN3_layer_call_and_return_conditional_losses_587961{*+:;,-./0123456789 <б9
2б/
)і&
input_1         KK
p 
ф "%б"
і
0         
џ ┐
@__inference_CNN3_layer_call_and_return_conditional_losses_588088{*+:;,-./0123456789 <б9
2б/
)і&
input_1         KK
p
ф "%б"
і
0         
џ Ќ
%__inference_CNN3_layer_call_fn_588133n*+:;,-./0123456789 <б9
2б/
)і&
input_1         KK
p 
ф "і         ќ
%__inference_CNN3_layer_call_fn_588178m*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p 
ф "і         ќ
%__inference_CNN3_layer_call_fn_588223m*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p
ф "і         Ќ
%__inference_CNN3_layer_call_fn_588268n*+:;,-./0123456789 <б9
2б/
)і&
input_1         KK
p
ф "і         Ф
!__inference__wrapped_model_586073Ё*+:;,-./0123456789 8б5
.б+
)і&
input_1         KK
ф "3ф0
.
output_1"і
output_1         В
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588952ќ*+:;MбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ В
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588970ќ*+:;MбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ К
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_588988r*+:;;б8
1б.
(і%
inputs         KK
p 
ф "-б*
#і 
0         KK
џ К
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_589006r*+:;;б8
1б.
(і%
inputs         KK
p
ф "-б*
#і 
0         KK
џ ─
6__inference_batch_normalization_4_layer_call_fn_589019Ѕ*+:;MбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ─
6__inference_batch_normalization_4_layer_call_fn_589032Ѕ*+:;MбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           Ъ
6__inference_batch_normalization_4_layer_call_fn_589045e*+:;;б8
1б.
(і%
inputs         KK
p 
ф " і         KKЪ
6__inference_batch_normalization_4_layer_call_fn_589058e*+:;;б8
1б.
(і%
inputs         KK
p
ф " і         KKx
__inference_call_532478]*+:;,-./0123456789 3б0
)б&
 і
inputsђKK
p
ф "і	ђx
__inference_call_532566]*+:;,-./0123456789 3б0
)б&
 і
inputsђKK
p 
ф "і	ђѕ
__inference_call_532654m*+:;,-./0123456789 ;б8
1б.
(і%
inputs         KK
p 
ф "і         х
E__inference_conv2d_20_layer_call_and_return_conditional_losses_589081l,-7б4
-б*
(і%
inputs         KK
ф "-б*
#і 
0         KK 
џ Ї
*__inference_conv2d_20_layer_call_fn_589090_,-7б4
-б*
(і%
inputs         KK
ф " і         KK Х
E__inference_conv2d_21_layer_call_and_return_conditional_losses_589101m./7б4
-б*
(і%
inputs         %% 
ф ".б+
$і!
0         %%ђ
џ ј
*__inference_conv2d_21_layer_call_fn_589110`./7б4
-б*
(і%
inputs         %% 
ф "!і         %%ђи
E__inference_conv2d_22_layer_call_and_return_conditional_losses_589121n018б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ Ј
*__inference_conv2d_22_layer_call_fn_589130a018б5
.б+
)і&
inputs         ђ
ф "!і         ђи
E__inference_conv2d_23_layer_call_and_return_conditional_losses_589141n238б5
.б+
)і&
inputs         		ђ
ф ".б+
$і!
0         		ђ
џ Ј
*__inference_conv2d_23_layer_call_fn_589150a238б5
.б+
)і&
inputs         		ђ
ф "!і         		ђи
E__inference_conv2d_24_layer_call_and_return_conditional_losses_589161n458б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ Ј
*__inference_conv2d_24_layer_call_fn_589170a458б5
.б+
)і&
inputs         ђ
ф "!і         ђд
D__inference_dense_12_layer_call_and_return_conditional_losses_589231^670б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ~
)__inference_dense_12_layer_call_fn_589240Q670б-
&б#
!і
inputs         ђ
ф "і         ђд
D__inference_dense_13_layer_call_and_return_conditional_losses_589290^890б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ~
)__inference_dense_13_layer_call_fn_589299Q890б-
&б#
!і
inputs         ђ
ф "і         ђЦ
D__inference_dense_14_layer_call_and_return_conditional_losses_588899] 0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ }
)__inference_dense_14_layer_call_fn_588908P 0б-
&б#
!і
inputs         ђ
ф "і         И
F__inference_dropout_12_layer_call_and_return_conditional_losses_589175n<б9
2б/
)і&
inputs         ђ
p 
ф ".б+
$і!
0         ђ
џ И
F__inference_dropout_12_layer_call_and_return_conditional_losses_589187n<б9
2б/
)і&
inputs         ђ
p
ф ".б+
$і!
0         ђ
џ љ
+__inference_dropout_12_layer_call_fn_589192a<б9
2б/
)і&
inputs         ђ
p 
ф "!і         ђљ
+__inference_dropout_12_layer_call_fn_589197a<б9
2б/
)і&
inputs         ђ
p
ф "!і         ђе
F__inference_dropout_13_layer_call_and_return_conditional_losses_589245^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_13_layer_call_and_return_conditional_losses_589257^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ђ
+__inference_dropout_13_layer_call_fn_589262Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђђ
+__inference_dropout_13_layer_call_fn_589267Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђе
F__inference_dropout_14_layer_call_and_return_conditional_losses_589304^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_14_layer_call_and_return_conditional_losses_589316^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ђ
+__inference_dropout_14_layer_call_fn_589321Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђђ
+__inference_dropout_14_layer_call_fn_589326Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђФ
E__inference_flatten_4_layer_call_and_return_conditional_losses_589203b8б5
.б+
)і&
inputs         ђ
ф "&б#
і
0         ђ
џ Ѓ
*__inference_flatten_4_layer_call_fn_589208U8б5
.б+
)і&
inputs         ђ
ф "і         ђИ
D__inference_lambda_4_layer_call_and_return_conditional_losses_588916p?б<
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
D__inference_lambda_4_layer_call_and_return_conditional_losses_588924p?б<
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
)__inference_lambda_4_layer_call_fn_588929c?б<
5б2
(і%
inputs         KK

 
p 
ф " і         KKљ
)__inference_lambda_4_layer_call_fn_588934c?б<
5б2
(і%
inputs         KK

 
p
ф " і         KK;
__inference_loss_fn_0_589337,б

б 
ф "і ;
__inference_loss_fn_1_5893486б

б 
ф "і ;
__inference_loss_fn_2_5893598б

б 
ф "і №
L__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_586205ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_20_layer_call_fn_586211ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_586217ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_21_layer_call_fn_586223ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_586229ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_22_layer_call_fn_586235ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_586241ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_23_layer_call_fn_586247ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_586253ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_24_layer_call_fn_586259ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ╔
H__inference_sequential_4_layer_call_and_return_conditional_losses_588385}*+:;,-./0123456789?б<
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_588505}*+:;,-./0123456789?б<
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_588604Ё*+:;,-./0123456789GбD
=б:
0і-
lambda_4_input         KK
p 

 
ф "&б#
і
0         ђ
џ м
H__inference_sequential_4_layer_call_and_return_conditional_losses_588724Ё*+:;,-./0123456789GбD
=б:
0і-
lambda_4_input         KK
p

 
ф "&б#
і
0         ђ
џ Е
-__inference_sequential_4_layer_call_fn_588765x*+:;,-./0123456789GбD
=б:
0і-
lambda_4_input         KK
p 

 
ф "і         ђА
-__inference_sequential_4_layer_call_fn_588806p*+:;,-./0123456789?б<
5б2
(і%
inputs         KK
p 

 
ф "і         ђА
-__inference_sequential_4_layer_call_fn_588847p*+:;,-./0123456789?б<
5б2
(і%
inputs         KK
p

 
ф "і         ђЕ
-__inference_sequential_4_layer_call_fn_588888x*+:;,-./0123456789GбD
=б:
0і-
lambda_4_input         KK
p

 
ф "і         ђ╣
$__inference_signature_wrapper_587622љ*+:;,-./0123456789 Cб@
б 
9ф6
4
input_1)і&
input_1         KK"3ф0
.
output_1"і
output_1         