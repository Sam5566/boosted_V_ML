аД:
╬Ю
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
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
Ы
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
·
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
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
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
В
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
delete_old_dirsbool(И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ег2
{
dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_44/kernel
t
#dense_44/kernel/Read/ReadVariableOpReadVariableOpdense_44/kernel*
_output_shapes
:	А*
dtype0
r
dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_44/bias
k
!dense_44/bias/Read/ReadVariableOpReadVariableOpdense_44/bias*
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
Р
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_16/gamma
Й
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_16/beta
З
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_16/moving_mean
Х
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:*
dtype0
д
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_16/moving_variance
Э
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:*
dtype0
Д
conv2d_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_48/kernel
}
$conv2d_48/kernel/Read/ReadVariableOpReadVariableOpconv2d_48/kernel*&
_output_shapes
: *
dtype0
t
conv2d_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_48/bias
m
"conv2d_48/bias/Read/ReadVariableOpReadVariableOpconv2d_48/bias*
_output_shapes
: *
dtype0
Е
conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*!
shared_nameconv2d_49/kernel
~
$conv2d_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_49/kernel*'
_output_shapes
: А*
dtype0
u
conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_49/bias
n
"conv2d_49/bias/Read/ReadVariableOpReadVariableOpconv2d_49/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_50/kernel

$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_50/bias
n
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
_output_shapes	
:А*
dtype0
}
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА* 
shared_namedense_40/kernel
v
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*!
_output_shapes
:АвА*
dtype0
s
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_40/bias
l
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes	
:А*
dtype0
|
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_41/kernel
u
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_41/bias
l
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes	
:А*
dtype0
Р
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_17/gamma
Й
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_17/beta
З
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_17/moving_mean
Х
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:*
dtype0
д
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_17/moving_variance
Э
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:*
dtype0
Д
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_51/kernel
}
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*&
_output_shapes
: *
dtype0
t
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes
: *
dtype0
Е
conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*!
shared_nameconv2d_52/kernel
~
$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*'
_output_shapes
: А*
dtype0
u
conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_52/bias
n
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_53/kernel

$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_53/bias
n
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes	
:А*
dtype0
}
dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА* 
shared_namedense_42/kernel
v
#dense_42/kernel/Read/ReadVariableOpReadVariableOpdense_42/kernel*!
_output_shapes
:АвА*
dtype0
s
dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_42/bias
l
!dense_42/bias/Read/ReadVariableOpReadVariableOpdense_42/bias*
_output_shapes	
:А*
dtype0
|
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_43/kernel
u
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_43/bias
l
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes	
:А*
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
Й
Adam/dense_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_44/kernel/m
В
*Adam/dense_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_44/kernel/m*
_output_shapes
:	А*
dtype0
А
Adam/dense_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_44/bias/m
y
(Adam/dense_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_44/bias/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_16/gamma/m
Ч
7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_16/beta/m
Х
6Adam/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_48/kernel/m
Л
+Adam/conv2d_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_48/bias/m
{
)Adam/conv2d_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/m*
_output_shapes
: *
dtype0
У
Adam/conv2d_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_49/kernel/m
М
+Adam/conv2d_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/m*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_49/bias/m
|
)Adam/conv2d_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_50/kernel/m
Н
+Adam/conv2d_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_50/bias/m
|
)Adam/conv2d_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/m*
_output_shapes	
:А*
dtype0
Л
Adam/dense_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*'
shared_nameAdam/dense_40/kernel/m
Д
*Adam/dense_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/m*!
_output_shapes
:АвА*
dtype0
Б
Adam/dense_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_40/bias/m
z
(Adam/dense_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_41/kernel/m
Г
*Adam/dense_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_41/bias/m
z
(Adam/dense_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/m*
_output_shapes	
:А*
dtype0
Ю
#Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_17/gamma/m
Ч
7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_17/beta/m
Х
6Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_51/kernel/m
Л
+Adam/conv2d_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_51/bias/m
{
)Adam/conv2d_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/m*
_output_shapes
: *
dtype0
У
Adam/conv2d_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_52/kernel/m
М
+Adam/conv2d_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/m*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_52/bias/m
|
)Adam/conv2d_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_53/kernel/m
Н
+Adam/conv2d_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_53/bias/m
|
)Adam/conv2d_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/m*
_output_shapes	
:А*
dtype0
Л
Adam/dense_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*'
shared_nameAdam/dense_42/kernel/m
Д
*Adam/dense_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/m*!
_output_shapes
:АвА*
dtype0
Б
Adam/dense_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_42/bias/m
z
(Adam/dense_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_43/kernel/m
Г
*Adam/dense_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_43/bias/m
z
(Adam/dense_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_44/kernel/v
В
*Adam/dense_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_44/kernel/v*
_output_shapes
:	А*
dtype0
А
Adam/dense_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_44/bias/v
y
(Adam/dense_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_44/bias/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_16/gamma/v
Ч
7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_16/beta/v
Х
6Adam/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_48/kernel/v
Л
+Adam/conv2d_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_48/bias/v
{
)Adam/conv2d_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/v*
_output_shapes
: *
dtype0
У
Adam/conv2d_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_49/kernel/v
М
+Adam/conv2d_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/v*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_49/bias/v
|
)Adam/conv2d_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_50/kernel/v
Н
+Adam/conv2d_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_50/bias/v
|
)Adam/conv2d_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/v*
_output_shapes	
:А*
dtype0
Л
Adam/dense_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*'
shared_nameAdam/dense_40/kernel/v
Д
*Adam/dense_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/v*!
_output_shapes
:АвА*
dtype0
Б
Adam/dense_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_40/bias/v
z
(Adam/dense_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_41/kernel/v
Г
*Adam/dense_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_41/bias/v
z
(Adam/dense_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/v*
_output_shapes	
:А*
dtype0
Ю
#Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_17/gamma/v
Ч
7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_17/beta/v
Х
6Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_51/kernel/v
Л
+Adam/conv2d_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_51/bias/v
{
)Adam/conv2d_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/v*
_output_shapes
: *
dtype0
У
Adam/conv2d_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_52/kernel/v
М
+Adam/conv2d_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/v*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_52/bias/v
|
)Adam/conv2d_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_53/kernel/v
Н
+Adam/conv2d_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_53/bias/v
|
)Adam/conv2d_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/v*
_output_shapes	
:А*
dtype0
Л
Adam/dense_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*'
shared_nameAdam/dense_42/kernel/v
Д
*Adam/dense_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/v*!
_output_shapes
:АвА*
dtype0
Б
Adam/dense_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_42/bias/v
z
(Adam/dense_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_43/kernel/v
Г
*Adam/dense_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_43/bias/v
z
(Adam/dense_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
жк
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*рй
value╒йB╤й B╔й
Ц

h2ptjl

h2ptj2
_output
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
и

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
и
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
 layer_with_weights-2
 layer-4
!layer-5
"layer_with_weights-3
"layer-6
#layer-7
$layer-8
%layer-9
&layer_with_weights-4
&layer-10
'layer-11
(layer_with_weights-5
(layer-12
)layer-13
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
╚
4iter

5beta_1

6beta_2
	7decay
8learning_rate.mЄ/mє9mЇ:mї=mЎ>mў?m°@m∙Am·Bm√Cm№Dm¤Em■Fm GmАHmБKmВLmГMmДNmЕOmЖPmЗQmИRmЙSmКTmЛ.vМ/vН9vО:vП=vР>vС?vТ@vУAvФBvХCvЦDvЧEvШFvЩGvЪHvЫKvЬLvЭMvЮNvЯOvаPvбQvвRvгSvдTvе
ц
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
.28
/29
╞
90
:1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
.24
/25
 
н
	variables
trainable_variables
Umetrics
Vlayer_regularization_losses
Wlayer_metrics

Xlayers
regularization_losses
Ynon_trainable_variables
 
R
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
Ч
^axis
	9gamma
:beta
;moving_mean
<moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
h

=kernel
>bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
R
g	variables
htrainable_variables
iregularization_losses
j	keras_api
h

?kernel
@bias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
R
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

Akernel
Bbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
R
{	variables
|trainable_variables
}regularization_losses
~	keras_api
U
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
l

Ckernel
Dbias
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
V
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
l

Ekernel
Fbias
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
V
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
f
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
V
90
:1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
 
▓
	variables
trainable_variables
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
Цlayers
regularization_losses
Чnon_trainable_variables
V
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь
	Ьaxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
l

Kkernel
Lbias
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
V
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
l

Mkernel
Nbias
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
V
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
l

Okernel
Pbias
▒	variables
▓trainable_variables
│regularization_losses
┤	keras_api
V
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
V
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
V
╜	variables
╛trainable_variables
┐regularization_losses
└	keras_api
l

Qkernel
Rbias
┴	variables
┬trainable_variables
├regularization_losses
─	keras_api
V
┼	variables
╞trainable_variables
╟regularization_losses
╚	keras_api
l

Skernel
Tbias
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
V
═	variables
╬trainable_variables
╧regularization_losses
╨	keras_api
f
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
V
G0
H1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
 
▓
*	variables
+trainable_variables
╤metrics
 ╥layer_regularization_losses
╙layer_metrics
╘layers
,regularization_losses
╒non_trainable_variables
NL
VARIABLE_VALUEdense_44/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_44/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
▓
0	variables
1trainable_variables
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
┘layers
2regularization_losses
┌non_trainable_variables
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
VARIABLE_VALUEconv2d_48/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_48/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_49/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_49/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_50/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_50/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_40/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_40/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_41/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_41/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_17/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_17/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_17/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_17/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_51/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_51/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_52/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_52/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_53/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_53/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_42/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_42/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_43/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_43/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE

█0
▄1
 
 

0
1
2

;0
<1
I2
J3
 
 
 
▓
Z	variables
[trainable_variables
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
рlayers
\regularization_losses
сnon_trainable_variables
 

90
:1
;2
<3

90
:1
 
▓
_	variables
`trainable_variables
тmetrics
 уlayer_regularization_losses
фlayer_metrics
хlayers
aregularization_losses
цnon_trainable_variables

=0
>1

=0
>1
 
▓
c	variables
dtrainable_variables
чmetrics
 шlayer_regularization_losses
щlayer_metrics
ъlayers
eregularization_losses
ыnon_trainable_variables
 
 
 
▓
g	variables
htrainable_variables
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
яlayers
iregularization_losses
Ёnon_trainable_variables

?0
@1

?0
@1
 
▓
k	variables
ltrainable_variables
ёmetrics
 Єlayer_regularization_losses
єlayer_metrics
Їlayers
mregularization_losses
їnon_trainable_variables
 
 
 
▓
o	variables
ptrainable_variables
Ўmetrics
 ўlayer_regularization_losses
°layer_metrics
∙layers
qregularization_losses
·non_trainable_variables

A0
B1

A0
B1
 
▓
s	variables
ttrainable_variables
√metrics
 №layer_regularization_losses
¤layer_metrics
■layers
uregularization_losses
 non_trainable_variables
 
 
 
▓
w	variables
xtrainable_variables
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
Гlayers
yregularization_losses
Дnon_trainable_variables
 
 
 
▓
{	variables
|trainable_variables
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
Иlayers
}regularization_losses
Йnon_trainable_variables
 
 
 
┤
	variables
Аtrainable_variables
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
Нlayers
Бregularization_losses
Оnon_trainable_variables

C0
D1

C0
D1
 
╡
Г	variables
Дtrainable_variables
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Тlayers
Еregularization_losses
Уnon_trainable_variables
 
 
 
╡
З	variables
Иtrainable_variables
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
Чlayers
Йregularization_losses
Шnon_trainable_variables

E0
F1

E0
F1
 
╡
Л	variables
Мtrainable_variables
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
Ьlayers
Нregularization_losses
Эnon_trainable_variables
 
 
 
╡
П	variables
Рtrainable_variables
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
бlayers
Сregularization_losses
вnon_trainable_variables
 
 
 
f

0
1
2
3
4
5
6
7
8
9
10
11
12
13

;0
<1
 
 
 
╡
Ш	variables
Щtrainable_variables
гmetrics
 дlayer_regularization_losses
еlayer_metrics
жlayers
Ъregularization_losses
зnon_trainable_variables
 

G0
H1
I2
J3

G0
H1
 
╡
Э	variables
Юtrainable_variables
иmetrics
 йlayer_regularization_losses
кlayer_metrics
лlayers
Яregularization_losses
мnon_trainable_variables

K0
L1

K0
L1
 
╡
б	variables
вtrainable_variables
нmetrics
 оlayer_regularization_losses
пlayer_metrics
░layers
гregularization_losses
▒non_trainable_variables
 
 
 
╡
е	variables
жtrainable_variables
▓metrics
 │layer_regularization_losses
┤layer_metrics
╡layers
зregularization_losses
╢non_trainable_variables

M0
N1

M0
N1
 
╡
й	variables
кtrainable_variables
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
║layers
лregularization_losses
╗non_trainable_variables
 
 
 
╡
н	variables
оtrainable_variables
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
┐layers
пregularization_losses
└non_trainable_variables

O0
P1

O0
P1
 
╡
▒	variables
▓trainable_variables
┴metrics
 ┬layer_regularization_losses
├layer_metrics
─layers
│regularization_losses
┼non_trainable_variables
 
 
 
╡
╡	variables
╢trainable_variables
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
╔layers
╖regularization_losses
╩non_trainable_variables
 
 
 
╡
╣	variables
║trainable_variables
╦metrics
 ╠layer_regularization_losses
═layer_metrics
╬layers
╗regularization_losses
╧non_trainable_variables
 
 
 
╡
╜	variables
╛trainable_variables
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
╙layers
┐regularization_losses
╘non_trainable_variables

Q0
R1

Q0
R1
 
╡
┴	variables
┬trainable_variables
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
╪layers
├regularization_losses
┘non_trainable_variables
 
 
 
╡
┼	variables
╞trainable_variables
┌metrics
 █layer_regularization_losses
▄layer_metrics
▌layers
╟regularization_losses
▐non_trainable_variables

S0
T1

S0
T1
 
╡
╔	variables
╩trainable_variables
▀metrics
 рlayer_regularization_losses
сlayer_metrics
тlayers
╦regularization_losses
уnon_trainable_variables
 
 
 
╡
═	variables
╬trainable_variables
фmetrics
 хlayer_regularization_losses
цlayer_metrics
чlayers
╧regularization_losses
шnon_trainable_variables
 
 
 
f
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12
)13

I0
J1
 
 
 
 
 
8

щtotal

ъcount
ы	variables
ь	keras_api
I

эtotal

юcount
я
_fn_kwargs
Ё	variables
ё	keras_api
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
I0
J1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
щ0
ъ1

ы	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

э0
ю1

Ё	variables
qo
VARIABLE_VALUEAdam/dense_44/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_44/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_16/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_48/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_48/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_49/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_49/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_50/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_50/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_40/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_40/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_41/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_41/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_17/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_51/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_51/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_52/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_52/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_53/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_53/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_42/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_42/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_43/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_43/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_44/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_44/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_16/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_48/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_48/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_49/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_49/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_50/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_50/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_40/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_40/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_41/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_41/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_17/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_51/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_51/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_52/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_52/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_53/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_53/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_42/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_42/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_43/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_43/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
ю
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_48/kernelconv2d_48/biasconv2d_49/kernelconv2d_49/biasconv2d_50/kernelconv2d_50/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/biasdense_44/kerneldense_44/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_2146187
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
з!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_44/kernel/Read/ReadVariableOp!dense_44/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp$conv2d_48/kernel/Read/ReadVariableOp"conv2d_48/bias/Read/ReadVariableOp$conv2d_49/kernel/Read/ReadVariableOp"conv2d_49/bias/Read/ReadVariableOp$conv2d_50/kernel/Read/ReadVariableOp"conv2d_50/bias/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp$conv2d_52/kernel/Read/ReadVariableOp"conv2d_52/bias/Read/ReadVariableOp$conv2d_53/kernel/Read/ReadVariableOp"conv2d_53/bias/Read/ReadVariableOp#dense_42/kernel/Read/ReadVariableOp!dense_42/bias/Read/ReadVariableOp#dense_43/kernel/Read/ReadVariableOp!dense_43/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_44/kernel/m/Read/ReadVariableOp(Adam/dense_44/bias/m/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_16/beta/m/Read/ReadVariableOp+Adam/conv2d_48/kernel/m/Read/ReadVariableOp)Adam/conv2d_48/bias/m/Read/ReadVariableOp+Adam/conv2d_49/kernel/m/Read/ReadVariableOp)Adam/conv2d_49/bias/m/Read/ReadVariableOp+Adam/conv2d_50/kernel/m/Read/ReadVariableOp)Adam/conv2d_50/bias/m/Read/ReadVariableOp*Adam/dense_40/kernel/m/Read/ReadVariableOp(Adam/dense_40/bias/m/Read/ReadVariableOp*Adam/dense_41/kernel/m/Read/ReadVariableOp(Adam/dense_41/bias/m/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_17/beta/m/Read/ReadVariableOp+Adam/conv2d_51/kernel/m/Read/ReadVariableOp)Adam/conv2d_51/bias/m/Read/ReadVariableOp+Adam/conv2d_52/kernel/m/Read/ReadVariableOp)Adam/conv2d_52/bias/m/Read/ReadVariableOp+Adam/conv2d_53/kernel/m/Read/ReadVariableOp)Adam/conv2d_53/bias/m/Read/ReadVariableOp*Adam/dense_42/kernel/m/Read/ReadVariableOp(Adam/dense_42/bias/m/Read/ReadVariableOp*Adam/dense_43/kernel/m/Read/ReadVariableOp(Adam/dense_43/bias/m/Read/ReadVariableOp*Adam/dense_44/kernel/v/Read/ReadVariableOp(Adam/dense_44/bias/v/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_16/beta/v/Read/ReadVariableOp+Adam/conv2d_48/kernel/v/Read/ReadVariableOp)Adam/conv2d_48/bias/v/Read/ReadVariableOp+Adam/conv2d_49/kernel/v/Read/ReadVariableOp)Adam/conv2d_49/bias/v/Read/ReadVariableOp+Adam/conv2d_50/kernel/v/Read/ReadVariableOp)Adam/conv2d_50/bias/v/Read/ReadVariableOp*Adam/dense_40/kernel/v/Read/ReadVariableOp(Adam/dense_40/bias/v/Read/ReadVariableOp*Adam/dense_41/kernel/v/Read/ReadVariableOp(Adam/dense_41/bias/v/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_17/beta/v/Read/ReadVariableOp+Adam/conv2d_51/kernel/v/Read/ReadVariableOp)Adam/conv2d_51/bias/v/Read/ReadVariableOp+Adam/conv2d_52/kernel/v/Read/ReadVariableOp)Adam/conv2d_52/bias/v/Read/ReadVariableOp+Adam/conv2d_53/kernel/v/Read/ReadVariableOp)Adam/conv2d_53/bias/v/Read/ReadVariableOp*Adam/dense_42/kernel/v/Read/ReadVariableOp(Adam/dense_42/bias/v/Read/ReadVariableOp*Adam/dense_43/kernel/v/Read/ReadVariableOp(Adam/dense_43/bias/v/Read/ReadVariableOpConst*h
Tina
_2]	*
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
GPU2 *0J 8В *)
f$R"
 __inference__traced_save_2149401
Ж
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_44/kerneldense_44/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_48/kernelconv2d_48/biasconv2d_49/kernelconv2d_49/biasconv2d_50/kernelconv2d_50/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/biastotalcounttotal_1count_1Adam/dense_44/kernel/mAdam/dense_44/bias/m#Adam/batch_normalization_16/gamma/m"Adam/batch_normalization_16/beta/mAdam/conv2d_48/kernel/mAdam/conv2d_48/bias/mAdam/conv2d_49/kernel/mAdam/conv2d_49/bias/mAdam/conv2d_50/kernel/mAdam/conv2d_50/bias/mAdam/dense_40/kernel/mAdam/dense_40/bias/mAdam/dense_41/kernel/mAdam/dense_41/bias/m#Adam/batch_normalization_17/gamma/m"Adam/batch_normalization_17/beta/mAdam/conv2d_51/kernel/mAdam/conv2d_51/bias/mAdam/conv2d_52/kernel/mAdam/conv2d_52/bias/mAdam/conv2d_53/kernel/mAdam/conv2d_53/bias/mAdam/dense_42/kernel/mAdam/dense_42/bias/mAdam/dense_43/kernel/mAdam/dense_43/bias/mAdam/dense_44/kernel/vAdam/dense_44/bias/v#Adam/batch_normalization_16/gamma/v"Adam/batch_normalization_16/beta/vAdam/conv2d_48/kernel/vAdam/conv2d_48/bias/vAdam/conv2d_49/kernel/vAdam/conv2d_49/bias/vAdam/conv2d_50/kernel/vAdam/conv2d_50/bias/vAdam/dense_40/kernel/vAdam/dense_40/bias/vAdam/dense_41/kernel/vAdam/dense_41/bias/v#Adam/batch_normalization_17/gamma/v"Adam/batch_normalization_17/beta/vAdam/conv2d_51/kernel/vAdam/conv2d_51/bias/vAdam/conv2d_52/kernel/vAdam/conv2d_52/bias/vAdam/conv2d_53/kernel/vAdam/conv2d_53/bias/vAdam/dense_42/kernel/vAdam/dense_42/bias/vAdam/dense_43/kernel/vAdam/dense_43/bias/v*g
Tin`
^2\*
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
GPU2 *0J 8В *,
f'R%
#__inference__traced_restore_2149684Я .
н
i
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2143778

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
°
e
G__inference_dropout_49_layer_call_and_return_conditional_losses_2148590

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Є
╙
8__inference_batch_normalization_16_layer_call_fn_2148322

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall╝
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_21436562
StatefulPartitionedCallи
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
Ы
╝
__inference_loss_fn_3_2149083U
;conv2d_51_kernel_regularizer_square_readvariableop_resource: 
identityИв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpь
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_51_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mulЬ
IdentityIdentity$conv2d_51/kernel/Regularizer/mul:z:03^conv2d_51/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp
┼^
╥
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2145740

inputs#
sequential_16_2145638:#
sequential_16_2145640:#
sequential_16_2145642:#
sequential_16_2145644:/
sequential_16_2145646: #
sequential_16_2145648: 0
sequential_16_2145650: А$
sequential_16_2145652:	А1
sequential_16_2145654:АА$
sequential_16_2145656:	А*
sequential_16_2145658:АвА$
sequential_16_2145660:	А)
sequential_16_2145662:
АА$
sequential_16_2145664:	А#
sequential_17_2145667:#
sequential_17_2145669:#
sequential_17_2145671:#
sequential_17_2145673:/
sequential_17_2145675: #
sequential_17_2145677: 0
sequential_17_2145679: А$
sequential_17_2145681:	А1
sequential_17_2145683:АА$
sequential_17_2145685:	А*
sequential_17_2145687:АвА$
sequential_17_2145689:	А)
sequential_17_2145691:
АА$
sequential_17_2145693:	А#
dense_44_2145698:	А
dense_44_2145700:
identityИв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpв dense_44/StatefulPartitionedCallв%sequential_16/StatefulPartitionedCallв%sequential_17/StatefulPartitionedCallр
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinputssequential_16_2145638sequential_16_2145640sequential_16_2145642sequential_16_2145644sequential_16_2145646sequential_16_2145648sequential_16_2145650sequential_16_2145652sequential_16_2145654sequential_16_2145656sequential_16_2145658sequential_16_2145660sequential_16_2145662sequential_16_2145664*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_21443122'
%sequential_16/StatefulPartitionedCallр
%sequential_17/StatefulPartitionedCallStatefulPartitionedCallinputssequential_17_2145667sequential_17_2145669sequential_17_2145671sequential_17_2145673sequential_17_2145675sequential_17_2145677sequential_17_2145679sequential_17_2145681sequential_17_2145683sequential_17_2145685sequential_17_2145687sequential_17_2145689sequential_17_2145691sequential_17_2145693*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_21451822'
%sequential_17/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╬
concatConcatV2.sequential_16/StatefulPartitionedCall:output:0.sequential_17/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatе
 dense_44/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_44_2145698dense_44_2145700*
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
GPU2 *0J 8В *N
fIRG
E__inference_dense_44_layer_call_and_return_conditional_losses_21454522"
 dense_44/StatefulPartitionedCall╞
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_2145646*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul┐
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_2145658*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╛
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_2145662* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul╞
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_2145675*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul┐
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_2145687*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╛
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_2145691* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulк
IdentityIdentity)dense_44/StatefulPartitionedCall:output:03^conv2d_48/kernel/Regularizer/Square/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp!^dense_44/StatefulPartitionedCall&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╢
f
G__inference_dropout_49_layer_call_and_return_conditional_losses_2148602

inputs
identityИc
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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
в
В
F__inference_conv2d_50_layer_call_and_return_conditional_losses_2143893

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
н
i
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2144636

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
║
н
E__inference_dense_41_layer_call_and_return_conditional_losses_2148634

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╟
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
є
И
/__inference_sequential_16_layer_call_fn_2147266
lambda_16_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCalllambda_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_21439942
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         KK
)
_user_specified_namelambda_16_input
└
┤
F__inference_conv2d_51_layer_call_and_return_conditional_losses_2148876

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu╧
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
°
e
G__inference_dropout_52_layer_call_and_return_conditional_losses_2144813

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
└
о
E__inference_dense_40_layer_call_and_return_conditional_losses_2143932

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpР
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╚
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
╢
f
G__inference_dropout_49_layer_call_and_return_conditional_losses_2144078

inputs
identityИc
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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Н
Ю
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2144526

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
ы
H
,__inference_dropout_51_layer_call_fn_2148921

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_51_layer_call_and_return_conditional_losses_21447752
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╔^
╥
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2145495

inputs#
sequential_16_2145381:#
sequential_16_2145383:#
sequential_16_2145385:#
sequential_16_2145387:/
sequential_16_2145389: #
sequential_16_2145391: 0
sequential_16_2145393: А$
sequential_16_2145395:	А1
sequential_16_2145397:АА$
sequential_16_2145399:	А*
sequential_16_2145401:АвА$
sequential_16_2145403:	А)
sequential_16_2145405:
АА$
sequential_16_2145407:	А#
sequential_17_2145410:#
sequential_17_2145412:#
sequential_17_2145414:#
sequential_17_2145416:/
sequential_17_2145418: #
sequential_17_2145420: 0
sequential_17_2145422: А$
sequential_17_2145424:	А1
sequential_17_2145426:АА$
sequential_17_2145428:	А*
sequential_17_2145430:АвА$
sequential_17_2145432:	А)
sequential_17_2145434:
АА$
sequential_17_2145436:	А#
dense_44_2145453:	А
dense_44_2145455:
identityИв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpв dense_44/StatefulPartitionedCallв%sequential_16/StatefulPartitionedCallв%sequential_17/StatefulPartitionedCallт
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinputssequential_16_2145381sequential_16_2145383sequential_16_2145385sequential_16_2145387sequential_16_2145389sequential_16_2145391sequential_16_2145393sequential_16_2145395sequential_16_2145397sequential_16_2145399sequential_16_2145401sequential_16_2145403sequential_16_2145405sequential_16_2145407*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_21439942'
%sequential_16/StatefulPartitionedCallт
%sequential_17/StatefulPartitionedCallStatefulPartitionedCallinputssequential_17_2145410sequential_17_2145412sequential_17_2145414sequential_17_2145416sequential_17_2145418sequential_17_2145420sequential_17_2145422sequential_17_2145424sequential_17_2145426sequential_17_2145428sequential_17_2145430sequential_17_2145432sequential_17_2145434sequential_17_2145436*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_21448642'
%sequential_17/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╬
concatConcatV2.sequential_16/StatefulPartitionedCall:output:0.sequential_17/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatе
 dense_44/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_44_2145453dense_44_2145455*
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
GPU2 *0J 8В *N
fIRG
E__inference_dense_44_layer_call_and_return_conditional_losses_21454522"
 dense_44/StatefulPartitionedCall╞
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_2145389*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul┐
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_2145401*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╛
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_2145405* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul╞
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_2145418*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul┐
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_2145430*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╛
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_2145434* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulк
IdentityIdentity)dense_44/StatefulPartitionedCall:output:03^conv2d_48/kernel/Regularizer/Square/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp!^dense_44/StatefulPartitionedCall&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
х
G
+__inference_lambda_16_layer_call_fn_2148293

inputs
identity╤
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
GPU2 *0J 8В *O
fJRH
F__inference_lambda_16_layer_call_and_return_conditional_losses_21442102
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
э
c
G__inference_flatten_16_layer_call_and_return_conditional_losses_2143913

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         Ав2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         Ав2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╣

ў
E__inference_dense_44_layer_call_and_return_conditional_losses_2145452

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╕
Ъ	
"__inference__wrapped_model_2143634
input_1
cnn_2jet_2143572:
cnn_2jet_2143574:
cnn_2jet_2143576:
cnn_2jet_2143578:*
cnn_2jet_2143580: 
cnn_2jet_2143582: +
cnn_2jet_2143584: А
cnn_2jet_2143586:	А,
cnn_2jet_2143588:АА
cnn_2jet_2143590:	А%
cnn_2jet_2143592:АвА
cnn_2jet_2143594:	А$
cnn_2jet_2143596:
АА
cnn_2jet_2143598:	А
cnn_2jet_2143600:
cnn_2jet_2143602:
cnn_2jet_2143604:
cnn_2jet_2143606:*
cnn_2jet_2143608: 
cnn_2jet_2143610: +
cnn_2jet_2143612: А
cnn_2jet_2143614:	А,
cnn_2jet_2143616:АА
cnn_2jet_2143618:	А%
cnn_2jet_2143620:АвА
cnn_2jet_2143622:	А$
cnn_2jet_2143624:
АА
cnn_2jet_2143626:	А#
cnn_2jet_2143628:	А
cnn_2jet_2143630:
identityИв CNN_2jet/StatefulPartitionedCallа
 CNN_2jet/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn_2jet_2143572cnn_2jet_2143574cnn_2jet_2143576cnn_2jet_2143578cnn_2jet_2143580cnn_2jet_2143582cnn_2jet_2143584cnn_2jet_2143586cnn_2jet_2143588cnn_2jet_2143590cnn_2jet_2143592cnn_2jet_2143594cnn_2jet_2143596cnn_2jet_2143598cnn_2jet_2143600cnn_2jet_2143602cnn_2jet_2143604cnn_2jet_2143606cnn_2jet_2143608cnn_2jet_2143610cnn_2jet_2143612cnn_2jet_2143614cnn_2jet_2143616cnn_2jet_2143618cnn_2jet_2143620cnn_2jet_2143622cnn_2jet_2143624cnn_2jet_2143626cnn_2jet_2143628cnn_2jet_2143630**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8В *!
fR
__inference_call_19850572"
 CNN_2jet/StatefulPartitionedCallа
IdentityIdentity)CNN_2jet/StatefulPartitionedCall:output:0!^CNN_2jet/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 CNN_2jet/StatefulPartitionedCall CNN_2jet/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
─
b
F__inference_lambda_17_layer_call_and_return_conditional_losses_2148712

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
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
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
м
╪
*__inference_CNN_2jet_layer_call_fn_2146252
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: %

unknown_19: А

unknown_20:	А&

unknown_21:АА

unknown_22:	А

unknown_23:АвА

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCall·
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
:         *@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_21454952
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
Ю
Б
F__inference_conv2d_52_layer_call_and_return_conditional_losses_2148896

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%А2

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
С 
Ж#
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2147215
input_1J
<sequential_16_batch_normalization_16_readvariableop_resource:L
>sequential_16_batch_normalization_16_readvariableop_1_resource:[
Msequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:]
Osequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_16_conv2d_48_conv2d_readvariableop_resource: E
7sequential_16_conv2d_48_biasadd_readvariableop_resource: Q
6sequential_16_conv2d_49_conv2d_readvariableop_resource: АF
7sequential_16_conv2d_49_biasadd_readvariableop_resource:	АR
6sequential_16_conv2d_50_conv2d_readvariableop_resource:ААF
7sequential_16_conv2d_50_biasadd_readvariableop_resource:	АJ
5sequential_16_dense_40_matmul_readvariableop_resource:АвАE
6sequential_16_dense_40_biasadd_readvariableop_resource:	АI
5sequential_16_dense_41_matmul_readvariableop_resource:
ААE
6sequential_16_dense_41_biasadd_readvariableop_resource:	АJ
<sequential_17_batch_normalization_17_readvariableop_resource:L
>sequential_17_batch_normalization_17_readvariableop_1_resource:[
Msequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_17_conv2d_51_conv2d_readvariableop_resource: E
7sequential_17_conv2d_51_biasadd_readvariableop_resource: Q
6sequential_17_conv2d_52_conv2d_readvariableop_resource: АF
7sequential_17_conv2d_52_biasadd_readvariableop_resource:	АR
6sequential_17_conv2d_53_conv2d_readvariableop_resource:ААF
7sequential_17_conv2d_53_biasadd_readvariableop_resource:	АJ
5sequential_17_dense_42_matmul_readvariableop_resource:АвАE
6sequential_17_dense_42_biasadd_readvariableop_resource:	АI
5sequential_17_dense_43_matmul_readvariableop_resource:
ААE
6sequential_17_dense_43_biasadd_readvariableop_resource:	А:
'dense_44_matmul_readvariableop_resource:	А6
(dense_44_biasadd_readvariableop_resource:
identityИв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpв3sequential_16/batch_normalization_16/AssignNewValueв5sequential_16/batch_normalization_16/AssignNewValue_1вDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpвFsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в3sequential_16/batch_normalization_16/ReadVariableOpв5sequential_16/batch_normalization_16/ReadVariableOp_1в.sequential_16/conv2d_48/BiasAdd/ReadVariableOpв-sequential_16/conv2d_48/Conv2D/ReadVariableOpв.sequential_16/conv2d_49/BiasAdd/ReadVariableOpв-sequential_16/conv2d_49/Conv2D/ReadVariableOpв.sequential_16/conv2d_50/BiasAdd/ReadVariableOpв-sequential_16/conv2d_50/Conv2D/ReadVariableOpв-sequential_16/dense_40/BiasAdd/ReadVariableOpв,sequential_16/dense_40/MatMul/ReadVariableOpв-sequential_16/dense_41/BiasAdd/ReadVariableOpв,sequential_16/dense_41/MatMul/ReadVariableOpв3sequential_17/batch_normalization_17/AssignNewValueв5sequential_17/batch_normalization_17/AssignNewValue_1вDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpвFsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в3sequential_17/batch_normalization_17/ReadVariableOpв5sequential_17/batch_normalization_17/ReadVariableOp_1в.sequential_17/conv2d_51/BiasAdd/ReadVariableOpв-sequential_17/conv2d_51/Conv2D/ReadVariableOpв.sequential_17/conv2d_52/BiasAdd/ReadVariableOpв-sequential_17/conv2d_52/Conv2D/ReadVariableOpв.sequential_17/conv2d_53/BiasAdd/ReadVariableOpв-sequential_17/conv2d_53/Conv2D/ReadVariableOpв-sequential_17/dense_42/BiasAdd/ReadVariableOpв,sequential_17/dense_42/MatMul/ReadVariableOpв-sequential_17/dense_43/BiasAdd/ReadVariableOpв,sequential_17/dense_43/MatMul/ReadVariableOp│
+sequential_16/lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_16/lambda_16/strided_slice/stack╖
-sequential_16/lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2/
-sequential_16/lambda_16/strided_slice/stack_1╖
-sequential_16/lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_16/lambda_16/strided_slice/stack_2Ў
%sequential_16/lambda_16/strided_sliceStridedSliceinput_14sequential_16/lambda_16/strided_slice/stack:output:06sequential_16/lambda_16/strided_slice/stack_1:output:06sequential_16/lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_16/lambda_16/strided_sliceу
3sequential_16/batch_normalization_16/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_16/batch_normalization_16/ReadVariableOpщ
5sequential_16/batch_normalization_16/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_16/batch_normalization_16/ReadVariableOp_1Ц
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1▐
5sequential_16/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3.sequential_16/lambda_16/strided_slice:output:0;sequential_16/batch_normalization_16/ReadVariableOp:value:0=sequential_16/batch_normalization_16/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<27
5sequential_16/batch_normalization_16/FusedBatchNormV3√
3sequential_16/batch_normalization_16/AssignNewValueAssignVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resourceBsequential_16/batch_normalization_16/FusedBatchNormV3:batch_mean:0E^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3sequential_16/batch_normalization_16/AssignNewValueЗ
5sequential_16/batch_normalization_16/AssignNewValue_1AssignVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resourceFsequential_16/batch_normalization_16/FusedBatchNormV3:batch_variance:0G^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5sequential_16/batch_normalization_16/AssignNewValue_1▌
-sequential_16/conv2d_48/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_48/Conv2D/ReadVariableOpЮ
sequential_16/conv2d_48/Conv2DConv2D9sequential_16/batch_normalization_16/FusedBatchNormV3:y:05sequential_16/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_16/conv2d_48/Conv2D╘
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpш
sequential_16/conv2d_48/BiasAddBiasAdd'sequential_16/conv2d_48/Conv2D:output:06sequential_16/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_16/conv2d_48/BiasAddи
sequential_16/conv2d_48/ReluRelu(sequential_16/conv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_16/conv2d_48/ReluЇ
&sequential_16/max_pooling2d_48/MaxPoolMaxPool*sequential_16/conv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool▐
-sequential_16/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_16/conv2d_49/Conv2D/ReadVariableOpХ
sequential_16/conv2d_49/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_16/conv2d_49/Conv2D╒
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_49/BiasAddBiasAdd'sequential_16/conv2d_49/Conv2D:output:06sequential_16/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_16/conv2d_49/BiasAddй
sequential_16/conv2d_49/ReluRelu(sequential_16/conv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_16/conv2d_49/Reluї
&sequential_16/max_pooling2d_49/MaxPoolMaxPool*sequential_16/conv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool▀
-sequential_16/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_16/conv2d_50/Conv2D/ReadVariableOpХ
sequential_16/conv2d_50/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:05sequential_16/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_16/conv2d_50/Conv2D╒
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_50/BiasAddBiasAdd'sequential_16/conv2d_50/Conv2D:output:06sequential_16/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_16/conv2d_50/BiasAddй
sequential_16/conv2d_50/ReluRelu(sequential_16/conv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_16/conv2d_50/Reluї
&sequential_16/max_pooling2d_50/MaxPoolMaxPool*sequential_16/conv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPoolХ
&sequential_16/dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2(
&sequential_16/dropout_48/dropout/ConstЁ
$sequential_16/dropout_48/dropout/MulMul/sequential_16/max_pooling2d_50/MaxPool:output:0/sequential_16/dropout_48/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2&
$sequential_16/dropout_48/dropout/Mulп
&sequential_16/dropout_48/dropout/ShapeShape/sequential_16/max_pooling2d_50/MaxPool:output:0*
T0*
_output_shapes
:2(
&sequential_16/dropout_48/dropout/ShapeИ
=sequential_16/dropout_48/dropout/random_uniform/RandomUniformRandomUniform/sequential_16/dropout_48/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02?
=sequential_16/dropout_48/dropout/random_uniform/RandomUniformз
/sequential_16/dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=21
/sequential_16/dropout_48/dropout/GreaterEqual/yл
-sequential_16/dropout_48/dropout/GreaterEqualGreaterEqualFsequential_16/dropout_48/dropout/random_uniform/RandomUniform:output:08sequential_16/dropout_48/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2/
-sequential_16/dropout_48/dropout/GreaterEqual╙
%sequential_16/dropout_48/dropout/CastCast1sequential_16/dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2'
%sequential_16/dropout_48/dropout/Castч
&sequential_16/dropout_48/dropout/Mul_1Mul(sequential_16/dropout_48/dropout/Mul:z:0)sequential_16/dropout_48/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2(
&sequential_16/dropout_48/dropout/Mul_1С
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_16/flatten_16/Const╪
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_48/dropout/Mul_1:z:0'sequential_16/flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_16/flatten_16/Reshape╒
,sequential_16/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_16/dense_40/MatMul/ReadVariableOp▄
sequential_16/dense_40/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/MatMul╥
-sequential_16/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_40/BiasAdd/ReadVariableOp▐
sequential_16/dense_40/BiasAddBiasAdd'sequential_16/dense_40/MatMul:product:05sequential_16/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_40/BiasAddЮ
sequential_16/dense_40/ReluRelu'sequential_16/dense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/ReluХ
&sequential_16/dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&sequential_16/dropout_49/dropout/Constт
$sequential_16/dropout_49/dropout/MulMul)sequential_16/dense_40/Relu:activations:0/sequential_16/dropout_49/dropout/Const:output:0*
T0*(
_output_shapes
:         А2&
$sequential_16/dropout_49/dropout/Mulй
&sequential_16/dropout_49/dropout/ShapeShape)sequential_16/dense_40/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_16/dropout_49/dropout/ShapeА
=sequential_16/dropout_49/dropout/random_uniform/RandomUniformRandomUniform/sequential_16/dropout_49/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02?
=sequential_16/dropout_49/dropout/random_uniform/RandomUniformз
/sequential_16/dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/sequential_16/dropout_49/dropout/GreaterEqual/yг
-sequential_16/dropout_49/dropout/GreaterEqualGreaterEqualFsequential_16/dropout_49/dropout/random_uniform/RandomUniform:output:08sequential_16/dropout_49/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2/
-sequential_16/dropout_49/dropout/GreaterEqual╦
%sequential_16/dropout_49/dropout/CastCast1sequential_16/dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2'
%sequential_16/dropout_49/dropout/Cast▀
&sequential_16/dropout_49/dropout/Mul_1Mul(sequential_16/dropout_49/dropout/Mul:z:0)sequential_16/dropout_49/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2(
&sequential_16/dropout_49/dropout/Mul_1╘
,sequential_16/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_16/dense_41/MatMul/ReadVariableOp▌
sequential_16/dense_41/MatMulMatMul*sequential_16/dropout_49/dropout/Mul_1:z:04sequential_16/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/MatMul╥
-sequential_16/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_41/BiasAdd/ReadVariableOp▐
sequential_16/dense_41/BiasAddBiasAdd'sequential_16/dense_41/MatMul:product:05sequential_16/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_41/BiasAddЮ
sequential_16/dense_41/ReluRelu'sequential_16/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/ReluХ
&sequential_16/dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&sequential_16/dropout_50/dropout/Constт
$sequential_16/dropout_50/dropout/MulMul)sequential_16/dense_41/Relu:activations:0/sequential_16/dropout_50/dropout/Const:output:0*
T0*(
_output_shapes
:         А2&
$sequential_16/dropout_50/dropout/Mulй
&sequential_16/dropout_50/dropout/ShapeShape)sequential_16/dense_41/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_16/dropout_50/dropout/ShapeА
=sequential_16/dropout_50/dropout/random_uniform/RandomUniformRandomUniform/sequential_16/dropout_50/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02?
=sequential_16/dropout_50/dropout/random_uniform/RandomUniformз
/sequential_16/dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/sequential_16/dropout_50/dropout/GreaterEqual/yг
-sequential_16/dropout_50/dropout/GreaterEqualGreaterEqualFsequential_16/dropout_50/dropout/random_uniform/RandomUniform:output:08sequential_16/dropout_50/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2/
-sequential_16/dropout_50/dropout/GreaterEqual╦
%sequential_16/dropout_50/dropout/CastCast1sequential_16/dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2'
%sequential_16/dropout_50/dropout/Cast▀
&sequential_16/dropout_50/dropout/Mul_1Mul(sequential_16/dropout_50/dropout/Mul:z:0)sequential_16/dropout_50/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2(
&sequential_16/dropout_50/dropout/Mul_1│
+sequential_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2-
+sequential_17/lambda_17/strided_slice/stack╖
-sequential_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2/
-sequential_17/lambda_17/strided_slice/stack_1╖
-sequential_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_17/lambda_17/strided_slice/stack_2Ў
%sequential_17/lambda_17/strided_sliceStridedSliceinput_14sequential_17/lambda_17/strided_slice/stack:output:06sequential_17/lambda_17/strided_slice/stack_1:output:06sequential_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_17/lambda_17/strided_sliceу
3sequential_17/batch_normalization_17/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_17/batch_normalization_17/ReadVariableOpщ
5sequential_17/batch_normalization_17/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_17/batch_normalization_17/ReadVariableOp_1Ц
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1▐
5sequential_17/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3.sequential_17/lambda_17/strided_slice:output:0;sequential_17/batch_normalization_17/ReadVariableOp:value:0=sequential_17/batch_normalization_17/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<27
5sequential_17/batch_normalization_17/FusedBatchNormV3√
3sequential_17/batch_normalization_17/AssignNewValueAssignVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resourceBsequential_17/batch_normalization_17/FusedBatchNormV3:batch_mean:0E^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3sequential_17/batch_normalization_17/AssignNewValueЗ
5sequential_17/batch_normalization_17/AssignNewValue_1AssignVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resourceFsequential_17/batch_normalization_17/FusedBatchNormV3:batch_variance:0G^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5sequential_17/batch_normalization_17/AssignNewValue_1▌
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOpЮ
sequential_17/conv2d_51/Conv2DConv2D9sequential_17/batch_normalization_17/FusedBatchNormV3:y:05sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_17/conv2d_51/Conv2D╘
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpш
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_17/conv2d_51/BiasAddи
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_17/conv2d_51/ReluЇ
&sequential_17/max_pooling2d_51/MaxPoolMaxPool*sequential_17/conv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_51/MaxPool▐
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOpХ
sequential_17/conv2d_52/Conv2DConv2D/sequential_17/max_pooling2d_51/MaxPool:output:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_17/conv2d_52/Conv2D╒
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_17/conv2d_52/BiasAddй
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_17/conv2d_52/Reluї
&sequential_17/max_pooling2d_52/MaxPoolMaxPool*sequential_17/conv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_52/MaxPool▀
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOpХ
sequential_17/conv2d_53/Conv2DConv2D/sequential_17/max_pooling2d_52/MaxPool:output:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_17/conv2d_53/Conv2D╒
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_17/conv2d_53/BiasAddй
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_17/conv2d_53/Reluї
&sequential_17/max_pooling2d_53/MaxPoolMaxPool*sequential_17/conv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_53/MaxPoolХ
&sequential_17/dropout_51/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2(
&sequential_17/dropout_51/dropout/ConstЁ
$sequential_17/dropout_51/dropout/MulMul/sequential_17/max_pooling2d_53/MaxPool:output:0/sequential_17/dropout_51/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2&
$sequential_17/dropout_51/dropout/Mulп
&sequential_17/dropout_51/dropout/ShapeShape/sequential_17/max_pooling2d_53/MaxPool:output:0*
T0*
_output_shapes
:2(
&sequential_17/dropout_51/dropout/ShapeИ
=sequential_17/dropout_51/dropout/random_uniform/RandomUniformRandomUniform/sequential_17/dropout_51/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02?
=sequential_17/dropout_51/dropout/random_uniform/RandomUniformз
/sequential_17/dropout_51/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=21
/sequential_17/dropout_51/dropout/GreaterEqual/yл
-sequential_17/dropout_51/dropout/GreaterEqualGreaterEqualFsequential_17/dropout_51/dropout/random_uniform/RandomUniform:output:08sequential_17/dropout_51/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2/
-sequential_17/dropout_51/dropout/GreaterEqual╙
%sequential_17/dropout_51/dropout/CastCast1sequential_17/dropout_51/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2'
%sequential_17/dropout_51/dropout/Castч
&sequential_17/dropout_51/dropout/Mul_1Mul(sequential_17/dropout_51/dropout/Mul:z:0)sequential_17/dropout_51/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2(
&sequential_17/dropout_51/dropout/Mul_1С
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_17/flatten_17/Const╪
 sequential_17/flatten_17/ReshapeReshape*sequential_17/dropout_51/dropout/Mul_1:z:0'sequential_17/flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_17/flatten_17/Reshape╒
,sequential_17/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_17/dense_42/MatMul/ReadVariableOp▄
sequential_17/dense_42/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/MatMul╥
-sequential_17/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_42/BiasAdd/ReadVariableOp▐
sequential_17/dense_42/BiasAddBiasAdd'sequential_17/dense_42/MatMul:product:05sequential_17/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_42/BiasAddЮ
sequential_17/dense_42/ReluRelu'sequential_17/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/ReluХ
&sequential_17/dropout_52/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&sequential_17/dropout_52/dropout/Constт
$sequential_17/dropout_52/dropout/MulMul)sequential_17/dense_42/Relu:activations:0/sequential_17/dropout_52/dropout/Const:output:0*
T0*(
_output_shapes
:         А2&
$sequential_17/dropout_52/dropout/Mulй
&sequential_17/dropout_52/dropout/ShapeShape)sequential_17/dense_42/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_17/dropout_52/dropout/ShapeА
=sequential_17/dropout_52/dropout/random_uniform/RandomUniformRandomUniform/sequential_17/dropout_52/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02?
=sequential_17/dropout_52/dropout/random_uniform/RandomUniformз
/sequential_17/dropout_52/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/sequential_17/dropout_52/dropout/GreaterEqual/yг
-sequential_17/dropout_52/dropout/GreaterEqualGreaterEqualFsequential_17/dropout_52/dropout/random_uniform/RandomUniform:output:08sequential_17/dropout_52/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2/
-sequential_17/dropout_52/dropout/GreaterEqual╦
%sequential_17/dropout_52/dropout/CastCast1sequential_17/dropout_52/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2'
%sequential_17/dropout_52/dropout/Cast▀
&sequential_17/dropout_52/dropout/Mul_1Mul(sequential_17/dropout_52/dropout/Mul:z:0)sequential_17/dropout_52/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2(
&sequential_17/dropout_52/dropout/Mul_1╘
,sequential_17/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_17/dense_43/MatMul/ReadVariableOp▌
sequential_17/dense_43/MatMulMatMul*sequential_17/dropout_52/dropout/Mul_1:z:04sequential_17/dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/MatMul╥
-sequential_17/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_43/BiasAdd/ReadVariableOp▐
sequential_17/dense_43/BiasAddBiasAdd'sequential_17/dense_43/MatMul:product:05sequential_17/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_43/BiasAddЮ
sequential_17/dense_43/ReluRelu'sequential_17/dense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/ReluХ
&sequential_17/dropout_53/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&sequential_17/dropout_53/dropout/Constт
$sequential_17/dropout_53/dropout/MulMul)sequential_17/dense_43/Relu:activations:0/sequential_17/dropout_53/dropout/Const:output:0*
T0*(
_output_shapes
:         А2&
$sequential_17/dropout_53/dropout/Mulй
&sequential_17/dropout_53/dropout/ShapeShape)sequential_17/dense_43/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_17/dropout_53/dropout/ShapeА
=sequential_17/dropout_53/dropout/random_uniform/RandomUniformRandomUniform/sequential_17/dropout_53/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02?
=sequential_17/dropout_53/dropout/random_uniform/RandomUniformз
/sequential_17/dropout_53/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/sequential_17/dropout_53/dropout/GreaterEqual/yг
-sequential_17/dropout_53/dropout/GreaterEqualGreaterEqualFsequential_17/dropout_53/dropout/random_uniform/RandomUniform:output:08sequential_17/dropout_53/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2/
-sequential_17/dropout_53/dropout/GreaterEqual╦
%sequential_17/dropout_53/dropout/CastCast1sequential_17/dropout_53/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2'
%sequential_17/dropout_53/dropout/Cast▀
&sequential_17/dropout_53/dropout/Mul_1Mul(sequential_17/dropout_53/dropout/Mul:z:0)sequential_17/dropout_53/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2(
&sequential_17/dropout_53/dropout/Mul_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╞
concatConcatV2*sequential_16/dropout_50/dropout/Mul_1:z:0*sequential_17/dropout_53/dropout/Mul_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatй
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_44/MatMul/ReadVariableOpЧ
dense_44/MatMulMatMulconcat:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpе
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/BiasAdd|
dense_44/SoftmaxSoftmaxdense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_44/Softmaxч
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul▀
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul▐
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulч
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul▀
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul▐
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulЕ
IdentityIdentitydense_44/Softmax:softmax:03^conv2d_48/kernel/Regularizer/Square/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp4^sequential_16/batch_normalization_16/AssignNewValue6^sequential_16/batch_normalization_16/AssignNewValue_1E^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_16/ReadVariableOp6^sequential_16/batch_normalization_16/ReadVariableOp_1/^sequential_16/conv2d_48/BiasAdd/ReadVariableOp.^sequential_16/conv2d_48/Conv2D/ReadVariableOp/^sequential_16/conv2d_49/BiasAdd/ReadVariableOp.^sequential_16/conv2d_49/Conv2D/ReadVariableOp/^sequential_16/conv2d_50/BiasAdd/ReadVariableOp.^sequential_16/conv2d_50/Conv2D/ReadVariableOp.^sequential_16/dense_40/BiasAdd/ReadVariableOp-^sequential_16/dense_40/MatMul/ReadVariableOp.^sequential_16/dense_41/BiasAdd/ReadVariableOp-^sequential_16/dense_41/MatMul/ReadVariableOp4^sequential_17/batch_normalization_17/AssignNewValue6^sequential_17/batch_normalization_17/AssignNewValue_1E^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_17/ReadVariableOp6^sequential_17/batch_normalization_17/ReadVariableOp_1/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_42/BiasAdd/ReadVariableOp-^sequential_17/dense_42/MatMul/ReadVariableOp.^sequential_17/dense_43/BiasAdd/ReadVariableOp-^sequential_17/dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2j
3sequential_16/batch_normalization_16/AssignNewValue3sequential_16/batch_normalization_16/AssignNewValue2n
5sequential_16/batch_normalization_16/AssignNewValue_15sequential_16/batch_normalization_16/AssignNewValue_12М
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_16/ReadVariableOp3sequential_16/batch_normalization_16/ReadVariableOp2n
5sequential_16/batch_normalization_16/ReadVariableOp_15sequential_16/batch_normalization_16/ReadVariableOp_12`
.sequential_16/conv2d_48/BiasAdd/ReadVariableOp.sequential_16/conv2d_48/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_48/Conv2D/ReadVariableOp-sequential_16/conv2d_48/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_49/BiasAdd/ReadVariableOp.sequential_16/conv2d_49/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_49/Conv2D/ReadVariableOp-sequential_16/conv2d_49/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_50/BiasAdd/ReadVariableOp.sequential_16/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_50/Conv2D/ReadVariableOp-sequential_16/conv2d_50/Conv2D/ReadVariableOp2^
-sequential_16/dense_40/BiasAdd/ReadVariableOp-sequential_16/dense_40/BiasAdd/ReadVariableOp2\
,sequential_16/dense_40/MatMul/ReadVariableOp,sequential_16/dense_40/MatMul/ReadVariableOp2^
-sequential_16/dense_41/BiasAdd/ReadVariableOp-sequential_16/dense_41/BiasAdd/ReadVariableOp2\
,sequential_16/dense_41/MatMul/ReadVariableOp,sequential_16/dense_41/MatMul/ReadVariableOp2j
3sequential_17/batch_normalization_17/AssignNewValue3sequential_17/batch_normalization_17/AssignNewValue2n
5sequential_17/batch_normalization_17/AssignNewValue_15sequential_17/batch_normalization_17/AssignNewValue_12М
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_17/ReadVariableOp3sequential_17/batch_normalization_17/ReadVariableOp2n
5sequential_17/batch_normalization_17/ReadVariableOp_15sequential_17/batch_normalization_17/ReadVariableOp_12`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_42/BiasAdd/ReadVariableOp-sequential_17/dense_42/BiasAdd/ReadVariableOp2\
,sequential_17/dense_42/MatMul/ReadVariableOp,sequential_17/dense_42/MatMul/ReadVariableOp2^
-sequential_17/dense_43/BiasAdd/ReadVariableOp-sequential_17/dense_43/BiasAdd/ReadVariableOp2\
,sequential_17/dense_43/MatMul/ReadVariableOp,sequential_17/dense_43/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
┼
Ю
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2143830

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
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
в
В
F__inference_conv2d_50_layer_call_and_return_conditional_losses_2148505

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
┼
Ю
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2144700

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
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
Н
Ю
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2143656

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
є
И
/__inference_sequential_17_layer_call_fn_2147790
lambda_17_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCalllambda_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_21448642
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         KK
)
_user_specified_namelambda_17_input
Ш
e
G__inference_dropout_48_layer_call_and_return_conditional_losses_2148520

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         		А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
Вт
┬
__inference_call_1988971

inputsJ
<sequential_16_batch_normalization_16_readvariableop_resource:L
>sequential_16_batch_normalization_16_readvariableop_1_resource:[
Msequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:]
Osequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_16_conv2d_48_conv2d_readvariableop_resource: E
7sequential_16_conv2d_48_biasadd_readvariableop_resource: Q
6sequential_16_conv2d_49_conv2d_readvariableop_resource: АF
7sequential_16_conv2d_49_biasadd_readvariableop_resource:	АR
6sequential_16_conv2d_50_conv2d_readvariableop_resource:ААF
7sequential_16_conv2d_50_biasadd_readvariableop_resource:	АJ
5sequential_16_dense_40_matmul_readvariableop_resource:АвАE
6sequential_16_dense_40_biasadd_readvariableop_resource:	АI
5sequential_16_dense_41_matmul_readvariableop_resource:
ААE
6sequential_16_dense_41_biasadd_readvariableop_resource:	АJ
<sequential_17_batch_normalization_17_readvariableop_resource:L
>sequential_17_batch_normalization_17_readvariableop_1_resource:[
Msequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_17_conv2d_51_conv2d_readvariableop_resource: E
7sequential_17_conv2d_51_biasadd_readvariableop_resource: Q
6sequential_17_conv2d_52_conv2d_readvariableop_resource: АF
7sequential_17_conv2d_52_biasadd_readvariableop_resource:	АR
6sequential_17_conv2d_53_conv2d_readvariableop_resource:ААF
7sequential_17_conv2d_53_biasadd_readvariableop_resource:	АJ
5sequential_17_dense_42_matmul_readvariableop_resource:АвАE
6sequential_17_dense_42_biasadd_readvariableop_resource:	АI
5sequential_17_dense_43_matmul_readvariableop_resource:
ААE
6sequential_17_dense_43_biasadd_readvariableop_resource:	А:
'dense_44_matmul_readvariableop_resource:	А6
(dense_44_biasadd_readvariableop_resource:
identityИвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpвDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpвFsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в3sequential_16/batch_normalization_16/ReadVariableOpв5sequential_16/batch_normalization_16/ReadVariableOp_1в.sequential_16/conv2d_48/BiasAdd/ReadVariableOpв-sequential_16/conv2d_48/Conv2D/ReadVariableOpв.sequential_16/conv2d_49/BiasAdd/ReadVariableOpв-sequential_16/conv2d_49/Conv2D/ReadVariableOpв.sequential_16/conv2d_50/BiasAdd/ReadVariableOpв-sequential_16/conv2d_50/Conv2D/ReadVariableOpв-sequential_16/dense_40/BiasAdd/ReadVariableOpв,sequential_16/dense_40/MatMul/ReadVariableOpв-sequential_16/dense_41/BiasAdd/ReadVariableOpв,sequential_16/dense_41/MatMul/ReadVariableOpвDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpвFsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в3sequential_17/batch_normalization_17/ReadVariableOpв5sequential_17/batch_normalization_17/ReadVariableOp_1в.sequential_17/conv2d_51/BiasAdd/ReadVariableOpв-sequential_17/conv2d_51/Conv2D/ReadVariableOpв.sequential_17/conv2d_52/BiasAdd/ReadVariableOpв-sequential_17/conv2d_52/Conv2D/ReadVariableOpв.sequential_17/conv2d_53/BiasAdd/ReadVariableOpв-sequential_17/conv2d_53/Conv2D/ReadVariableOpв-sequential_17/dense_42/BiasAdd/ReadVariableOpв,sequential_17/dense_42/MatMul/ReadVariableOpв-sequential_17/dense_43/BiasAdd/ReadVariableOpв,sequential_17/dense_43/MatMul/ReadVariableOp│
+sequential_16/lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_16/lambda_16/strided_slice/stack╖
-sequential_16/lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2/
-sequential_16/lambda_16/strided_slice/stack_1╖
-sequential_16/lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_16/lambda_16/strided_slice/stack_2э
%sequential_16/lambda_16/strided_sliceStridedSliceinputs4sequential_16/lambda_16/strided_slice/stack:output:06sequential_16/lambda_16/strided_slice/stack_1:output:06sequential_16/lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2'
%sequential_16/lambda_16/strided_sliceу
3sequential_16/batch_normalization_16/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_16/batch_normalization_16/ReadVariableOpщ
5sequential_16/batch_normalization_16/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_16/batch_normalization_16/ReadVariableOp_1Ц
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1╚
5sequential_16/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3.sequential_16/lambda_16/strided_slice:output:0;sequential_16/batch_normalization_16/ReadVariableOp:value:0=sequential_16/batch_normalization_16/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 27
5sequential_16/batch_normalization_16/FusedBatchNormV3▌
-sequential_16/conv2d_48/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_48/Conv2D/ReadVariableOpЦ
sequential_16/conv2d_48/Conv2DConv2D9sequential_16/batch_normalization_16/FusedBatchNormV3:y:05sequential_16/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2 
sequential_16/conv2d_48/Conv2D╘
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpр
sequential_16/conv2d_48/BiasAddBiasAdd'sequential_16/conv2d_48/Conv2D:output:06sequential_16/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2!
sequential_16/conv2d_48/BiasAddа
sequential_16/conv2d_48/ReluRelu(sequential_16/conv2d_48/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_16/conv2d_48/Reluь
&sequential_16/max_pooling2d_48/MaxPoolMaxPool*sequential_16/conv2d_48/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool▐
-sequential_16/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_16/conv2d_49/Conv2D/ReadVariableOpН
sequential_16/conv2d_49/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2 
sequential_16/conv2d_49/Conv2D╒
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpс
sequential_16/conv2d_49/BiasAddBiasAdd'sequential_16/conv2d_49/Conv2D:output:06sequential_16/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2!
sequential_16/conv2d_49/BiasAddб
sequential_16/conv2d_49/ReluRelu(sequential_16/conv2d_49/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_16/conv2d_49/Reluэ
&sequential_16/max_pooling2d_49/MaxPoolMaxPool*sequential_16/conv2d_49/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool▀
-sequential_16/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_16/conv2d_50/Conv2D/ReadVariableOpН
sequential_16/conv2d_50/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:05sequential_16/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2 
sequential_16/conv2d_50/Conv2D╒
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpс
sequential_16/conv2d_50/BiasAddBiasAdd'sequential_16/conv2d_50/Conv2D:output:06sequential_16/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2!
sequential_16/conv2d_50/BiasAddб
sequential_16/conv2d_50/ReluRelu(sequential_16/conv2d_50/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_16/conv2d_50/Reluэ
&sequential_16/max_pooling2d_50/MaxPoolMaxPool*sequential_16/conv2d_50/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPool╢
!sequential_16/dropout_48/IdentityIdentity/sequential_16/max_pooling2d_50/MaxPool:output:0*
T0*(
_output_shapes
:А		А2#
!sequential_16/dropout_48/IdentityС
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_16/flatten_16/Const╨
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_48/Identity:output:0'sequential_16/flatten_16/Const:output:0*
T0*!
_output_shapes
:ААв2"
 sequential_16/flatten_16/Reshape╒
,sequential_16/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_16/dense_40/MatMul/ReadVariableOp╘
sequential_16/dense_40/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_40/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_16/dense_40/MatMul╥
-sequential_16/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_40/BiasAdd/ReadVariableOp╓
sequential_16/dense_40/BiasAddBiasAdd'sequential_16/dense_40/MatMul:product:05sequential_16/dense_40/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2 
sequential_16/dense_40/BiasAddЦ
sequential_16/dense_40/ReluRelu'sequential_16/dense_40/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_16/dense_40/Reluи
!sequential_16/dropout_49/IdentityIdentity)sequential_16/dense_40/Relu:activations:0*
T0* 
_output_shapes
:
АА2#
!sequential_16/dropout_49/Identity╘
,sequential_16/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_16/dense_41/MatMul/ReadVariableOp╒
sequential_16/dense_41/MatMulMatMul*sequential_16/dropout_49/Identity:output:04sequential_16/dense_41/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_16/dense_41/MatMul╥
-sequential_16/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_41/BiasAdd/ReadVariableOp╓
sequential_16/dense_41/BiasAddBiasAdd'sequential_16/dense_41/MatMul:product:05sequential_16/dense_41/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2 
sequential_16/dense_41/BiasAddЦ
sequential_16/dense_41/ReluRelu'sequential_16/dense_41/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_16/dense_41/Reluи
!sequential_16/dropout_50/IdentityIdentity)sequential_16/dense_41/Relu:activations:0*
T0* 
_output_shapes
:
АА2#
!sequential_16/dropout_50/Identity│
+sequential_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2-
+sequential_17/lambda_17/strided_slice/stack╖
-sequential_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2/
-sequential_17/lambda_17/strided_slice/stack_1╖
-sequential_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_17/lambda_17/strided_slice/stack_2э
%sequential_17/lambda_17/strided_sliceStridedSliceinputs4sequential_17/lambda_17/strided_slice/stack:output:06sequential_17/lambda_17/strided_slice/stack_1:output:06sequential_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2'
%sequential_17/lambda_17/strided_sliceу
3sequential_17/batch_normalization_17/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_17/batch_normalization_17/ReadVariableOpщ
5sequential_17/batch_normalization_17/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_17/batch_normalization_17/ReadVariableOp_1Ц
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1╚
5sequential_17/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3.sequential_17/lambda_17/strided_slice:output:0;sequential_17/batch_normalization_17/ReadVariableOp:value:0=sequential_17/batch_normalization_17/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 27
5sequential_17/batch_normalization_17/FusedBatchNormV3▌
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOpЦ
sequential_17/conv2d_51/Conv2DConv2D9sequential_17/batch_normalization_17/FusedBatchNormV3:y:05sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2 
sequential_17/conv2d_51/Conv2D╘
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpр
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2!
sequential_17/conv2d_51/BiasAddа
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_17/conv2d_51/Reluь
&sequential_17/max_pooling2d_51/MaxPoolMaxPool*sequential_17/conv2d_51/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_51/MaxPool▐
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOpН
sequential_17/conv2d_52/Conv2DConv2D/sequential_17/max_pooling2d_51/MaxPool:output:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2 
sequential_17/conv2d_52/Conv2D╒
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpс
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2!
sequential_17/conv2d_52/BiasAddб
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_17/conv2d_52/Reluэ
&sequential_17/max_pooling2d_52/MaxPoolMaxPool*sequential_17/conv2d_52/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_52/MaxPool▀
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOpН
sequential_17/conv2d_53/Conv2DConv2D/sequential_17/max_pooling2d_52/MaxPool:output:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2 
sequential_17/conv2d_53/Conv2D╒
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpс
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2!
sequential_17/conv2d_53/BiasAddб
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_17/conv2d_53/Reluэ
&sequential_17/max_pooling2d_53/MaxPoolMaxPool*sequential_17/conv2d_53/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_53/MaxPool╢
!sequential_17/dropout_51/IdentityIdentity/sequential_17/max_pooling2d_53/MaxPool:output:0*
T0*(
_output_shapes
:А		А2#
!sequential_17/dropout_51/IdentityС
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_17/flatten_17/Const╨
 sequential_17/flatten_17/ReshapeReshape*sequential_17/dropout_51/Identity:output:0'sequential_17/flatten_17/Const:output:0*
T0*!
_output_shapes
:ААв2"
 sequential_17/flatten_17/Reshape╒
,sequential_17/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_17/dense_42/MatMul/ReadVariableOp╘
sequential_17/dense_42/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_42/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_17/dense_42/MatMul╥
-sequential_17/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_42/BiasAdd/ReadVariableOp╓
sequential_17/dense_42/BiasAddBiasAdd'sequential_17/dense_42/MatMul:product:05sequential_17/dense_42/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2 
sequential_17/dense_42/BiasAddЦ
sequential_17/dense_42/ReluRelu'sequential_17/dense_42/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_17/dense_42/Reluи
!sequential_17/dropout_52/IdentityIdentity)sequential_17/dense_42/Relu:activations:0*
T0* 
_output_shapes
:
АА2#
!sequential_17/dropout_52/Identity╘
,sequential_17/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_17/dense_43/MatMul/ReadVariableOp╒
sequential_17/dense_43/MatMulMatMul*sequential_17/dropout_52/Identity:output:04sequential_17/dense_43/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_17/dense_43/MatMul╥
-sequential_17/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_43/BiasAdd/ReadVariableOp╓
sequential_17/dense_43/BiasAddBiasAdd'sequential_17/dense_43/MatMul:product:05sequential_17/dense_43/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2 
sequential_17/dense_43/BiasAddЦ
sequential_17/dense_43/ReluRelu'sequential_17/dense_43/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_17/dense_43/Reluи
!sequential_17/dropout_53/IdentityIdentity)sequential_17/dense_43/Relu:activations:0*
T0* 
_output_shapes
:
АА2#
!sequential_17/dropout_53/Identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╛
concatConcatV2*sequential_16/dropout_50/Identity:output:0*sequential_17/dropout_53/Identity:output:0concat/axis:output:0*
N*
T0* 
_output_shapes
:
АА2
concatй
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_44/MatMul/ReadVariableOpП
dense_44/MatMulMatMulconcat:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpЭ
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_44/BiasAddt
dense_44/SoftmaxSoftmaxdense_44/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_44/Softmaxч
IdentityIdentitydense_44/Softmax:softmax:0 ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOpE^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_16/ReadVariableOp6^sequential_16/batch_normalization_16/ReadVariableOp_1/^sequential_16/conv2d_48/BiasAdd/ReadVariableOp.^sequential_16/conv2d_48/Conv2D/ReadVariableOp/^sequential_16/conv2d_49/BiasAdd/ReadVariableOp.^sequential_16/conv2d_49/Conv2D/ReadVariableOp/^sequential_16/conv2d_50/BiasAdd/ReadVariableOp.^sequential_16/conv2d_50/Conv2D/ReadVariableOp.^sequential_16/dense_40/BiasAdd/ReadVariableOp-^sequential_16/dense_40/MatMul/ReadVariableOp.^sequential_16/dense_41/BiasAdd/ReadVariableOp-^sequential_16/dense_41/MatMul/ReadVariableOpE^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_17/ReadVariableOp6^sequential_17/batch_normalization_17/ReadVariableOp_1/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_42/BiasAdd/ReadVariableOp-^sequential_17/dense_42/MatMul/ReadVariableOp.^sequential_17/dense_43/BiasAdd/ReadVariableOp-^sequential_17/dense_43/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:АKK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2М
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_16/ReadVariableOp3sequential_16/batch_normalization_16/ReadVariableOp2n
5sequential_16/batch_normalization_16/ReadVariableOp_15sequential_16/batch_normalization_16/ReadVariableOp_12`
.sequential_16/conv2d_48/BiasAdd/ReadVariableOp.sequential_16/conv2d_48/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_48/Conv2D/ReadVariableOp-sequential_16/conv2d_48/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_49/BiasAdd/ReadVariableOp.sequential_16/conv2d_49/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_49/Conv2D/ReadVariableOp-sequential_16/conv2d_49/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_50/BiasAdd/ReadVariableOp.sequential_16/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_50/Conv2D/ReadVariableOp-sequential_16/conv2d_50/Conv2D/ReadVariableOp2^
-sequential_16/dense_40/BiasAdd/ReadVariableOp-sequential_16/dense_40/BiasAdd/ReadVariableOp2\
,sequential_16/dense_40/MatMul/ReadVariableOp,sequential_16/dense_40/MatMul/ReadVariableOp2^
-sequential_16/dense_41/BiasAdd/ReadVariableOp-sequential_16/dense_41/BiasAdd/ReadVariableOp2\
,sequential_16/dense_41/MatMul/ReadVariableOp,sequential_16/dense_41/MatMul/ReadVariableOp2М
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_17/ReadVariableOp3sequential_17/batch_normalization_17/ReadVariableOp2n
5sequential_17/batch_normalization_17/ReadVariableOp_15sequential_17/batch_normalization_17/ReadVariableOp_12`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_42/BiasAdd/ReadVariableOp-sequential_17/dense_42/BiasAdd/ReadVariableOp2\
,sequential_17/dense_42/MatMul/ReadVariableOp,sequential_17/dense_42/MatMul/ReadVariableOp2^
-sequential_17/dense_43/BiasAdd/ReadVariableOp-sequential_17/dense_43/BiasAdd/ReadVariableOp2\
,sequential_17/dense_43/MatMul/ReadVariableOp,sequential_17/dense_43/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
─
b
F__inference_lambda_17_layer_call_and_return_conditional_losses_2144681

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
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
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╪
 
/__inference_sequential_17_layer_call_fn_2147823

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallЯ
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
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_21448642
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ю
Б
F__inference_conv2d_49_layer_call_and_return_conditional_losses_2148485

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%А2

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
н
i
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2143790

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
°
e
G__inference_dropout_50_layer_call_and_return_conditional_losses_2148649

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
м
Ы
*__inference_dense_40_layer_call_fn_2148558

inputs
unknown:АвА
	unknown_0:	А
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_21439322
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
▀v
▒
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147635
lambda_16_input<
.batch_normalization_16_readvariableop_resource:>
0batch_normalization_16_readvariableop_1_resource:M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_48_conv2d_readvariableop_resource: 7
)conv2d_48_biasadd_readvariableop_resource: C
(conv2d_49_conv2d_readvariableop_resource: А8
)conv2d_49_biasadd_readvariableop_resource:	АD
(conv2d_50_conv2d_readvariableop_resource:АА8
)conv2d_50_biasadd_readvariableop_resource:	А<
'dense_40_matmul_readvariableop_resource:АвА7
(dense_40_biasadd_readvariableop_resource:	А;
'dense_41_matmul_readvariableop_resource:
АА7
(dense_41_biasadd_readvariableop_resource:	А
identityИв6batch_normalization_16/FusedBatchNormV3/ReadVariableOpв8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_16/ReadVariableOpв'batch_normalization_16/ReadVariableOp_1в conv2d_48/BiasAdd/ReadVariableOpвconv2d_48/Conv2D/ReadVariableOpв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв conv2d_49/BiasAdd/ReadVariableOpвconv2d_49/Conv2D/ReadVariableOpв conv2d_50/BiasAdd/ReadVariableOpвconv2d_50/Conv2D/ReadVariableOpвdense_40/BiasAdd/ReadVariableOpвdense_40/MatMul/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpвdense_41/BiasAdd/ReadVariableOpвdense_41/MatMul/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpЧ
lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_16/strided_slice/stackЫ
lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2!
lambda_16/strided_slice/stack_1Ы
lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2!
lambda_16/strided_slice/stack_2╕
lambda_16/strided_sliceStridedSlicelambda_16_input&lambda_16/strided_slice/stack:output:0(lambda_16/strided_slice/stack_1:output:0(lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_16/strided_slice╣
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp┐
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1ь
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ю
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 lambda_16/strided_slice:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3│
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_48/Conv2D/ReadVariableOpц
conv2d_48/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_48/Conv2Dк
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp░
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_48/BiasAdd~
conv2d_48/ReluReluconv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_48/Relu╩
max_pooling2d_48/MaxPoolMaxPoolconv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_48/MaxPool┤
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_49/Conv2D/ReadVariableOp▌
conv2d_49/Conv2DConv2D!max_pooling2d_48/MaxPool:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_49/Conv2Dл
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp▒
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_49/BiasAdd
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_49/Relu╦
max_pooling2d_49/MaxPoolMaxPoolconv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool╡
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_50/Conv2D/ReadVariableOp▌
conv2d_50/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_50/Conv2Dл
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp▒
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_50/BiasAdd
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_50/Relu╦
max_pooling2d_50/MaxPoolMaxPoolconv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPoolФ
dropout_48/IdentityIdentity!max_pooling2d_50/MaxPool:output:0*
T0*0
_output_shapes
:         		А2
dropout_48/Identityu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_16/Constа
flatten_16/ReshapeReshapedropout_48/Identity:output:0flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_16/Reshapeл
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02 
dense_40/MatMul/ReadVariableOpд
dense_40/MatMulMatMulflatten_16/Reshape:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_40/MatMulи
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_40/BiasAdd/ReadVariableOpж
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_40/BiasAddt
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_40/ReluЖ
dropout_49/IdentityIdentitydense_40/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_49/Identityк
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_41/MatMul/ReadVariableOpе
dense_41/MatMulMatMuldropout_49/Identity:output:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_41/MatMulи
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_41/BiasAdd/ReadVariableOpж
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_41/BiasAddt
dense_41/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_41/ReluЖ
dropout_50/IdentityIdentitydense_41/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_50/Identity┘
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul╤
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╨
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulй
IdentityIdentitydropout_50/Identity:output:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp3^conv2d_48/kernel/Regularizer/Square/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:` \
/
_output_shapes
:         KK
)
_user_specified_namelambda_16_input
─v
и
J__inference_sequential_17_layer_call_and_return_conditional_losses_2147972

inputs<
.batch_normalization_17_readvariableop_resource:>
0batch_normalization_17_readvariableop_1_resource:M
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_51_conv2d_readvariableop_resource: 7
)conv2d_51_biasadd_readvariableop_resource: C
(conv2d_52_conv2d_readvariableop_resource: А8
)conv2d_52_biasadd_readvariableop_resource:	АD
(conv2d_53_conv2d_readvariableop_resource:АА8
)conv2d_53_biasadd_readvariableop_resource:	А<
'dense_42_matmul_readvariableop_resource:АвА7
(dense_42_biasadd_readvariableop_resource:	А;
'dense_43_matmul_readvariableop_resource:
АА7
(dense_43_biasadd_readvariableop_resource:	А
identityИв6batch_normalization_17/FusedBatchNormV3/ReadVariableOpв8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_17/ReadVariableOpв'batch_normalization_17/ReadVariableOp_1в conv2d_51/BiasAdd/ReadVariableOpвconv2d_51/Conv2D/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв conv2d_52/BiasAdd/ReadVariableOpвconv2d_52/Conv2D/ReadVariableOpв conv2d_53/BiasAdd/ReadVariableOpвconv2d_53/Conv2D/ReadVariableOpвdense_42/BiasAdd/ReadVariableOpвdense_42/MatMul/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpвdense_43/BiasAdd/ReadVariableOpвdense_43/MatMul/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpЧ
lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
lambda_17/strided_slice/stackЫ
lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2!
lambda_17/strided_slice/stack_1Ы
lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2!
lambda_17/strided_slice/stack_2п
lambda_17/strided_sliceStridedSliceinputs&lambda_17/strided_slice/stack:output:0(lambda_17/strided_slice/stack_1:output:0(lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_17/strided_slice╣
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_17/ReadVariableOp┐
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_17/ReadVariableOp_1ь
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ю
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 lambda_17/strided_slice:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3│
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOpц
conv2d_51/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_51/Conv2Dк
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp░
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_51/Relu╩
max_pooling2d_51/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_51/MaxPool┤
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_52/Conv2D/ReadVariableOp▌
conv2d_52/Conv2DConv2D!max_pooling2d_51/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_52/Conv2Dл
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp▒
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_52/BiasAdd
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_52/Relu╦
max_pooling2d_52/MaxPoolMaxPoolconv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_52/MaxPool╡
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_53/Conv2D/ReadVariableOp▌
conv2d_53/Conv2DConv2D!max_pooling2d_52/MaxPool:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_53/Conv2Dл
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp▒
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_53/Relu╦
max_pooling2d_53/MaxPoolMaxPoolconv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_53/MaxPoolФ
dropout_51/IdentityIdentity!max_pooling2d_53/MaxPool:output:0*
T0*0
_output_shapes
:         		А2
dropout_51/Identityu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_17/Constа
flatten_17/ReshapeReshapedropout_51/Identity:output:0flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_17/Reshapeл
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02 
dense_42/MatMul/ReadVariableOpд
dense_42/MatMulMatMulflatten_17/Reshape:output:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_42/MatMulи
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_42/BiasAdd/ReadVariableOpж
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_42/BiasAddt
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_42/ReluЖ
dropout_52/IdentityIdentitydense_42/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_52/Identityк
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_43/MatMul/ReadVariableOpе
dense_43/MatMulMatMuldropout_52/Identity:output:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_43/MatMulи
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_43/BiasAdd/ReadVariableOpж
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_43/BiasAddt
dense_43/ReluReludense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_43/ReluЖ
dropout_53/IdentityIdentitydense_43/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_53/Identity┘
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul╤
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╨
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulй
IdentityIdentitydropout_53/Identity:output:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
и
╙
8__inference_batch_normalization_17_layer_call_fn_2148772

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallи
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_21450532
StatefulPartitionedCallЦ
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
Ю
Б
F__inference_conv2d_49_layer_call_and_return_conditional_losses_2143875

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%А2

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
╙г
к!
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2147002
input_1J
<sequential_16_batch_normalization_16_readvariableop_resource:L
>sequential_16_batch_normalization_16_readvariableop_1_resource:[
Msequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:]
Osequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_16_conv2d_48_conv2d_readvariableop_resource: E
7sequential_16_conv2d_48_biasadd_readvariableop_resource: Q
6sequential_16_conv2d_49_conv2d_readvariableop_resource: АF
7sequential_16_conv2d_49_biasadd_readvariableop_resource:	АR
6sequential_16_conv2d_50_conv2d_readvariableop_resource:ААF
7sequential_16_conv2d_50_biasadd_readvariableop_resource:	АJ
5sequential_16_dense_40_matmul_readvariableop_resource:АвАE
6sequential_16_dense_40_biasadd_readvariableop_resource:	АI
5sequential_16_dense_41_matmul_readvariableop_resource:
ААE
6sequential_16_dense_41_biasadd_readvariableop_resource:	АJ
<sequential_17_batch_normalization_17_readvariableop_resource:L
>sequential_17_batch_normalization_17_readvariableop_1_resource:[
Msequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_17_conv2d_51_conv2d_readvariableop_resource: E
7sequential_17_conv2d_51_biasadd_readvariableop_resource: Q
6sequential_17_conv2d_52_conv2d_readvariableop_resource: АF
7sequential_17_conv2d_52_biasadd_readvariableop_resource:	АR
6sequential_17_conv2d_53_conv2d_readvariableop_resource:ААF
7sequential_17_conv2d_53_biasadd_readvariableop_resource:	АJ
5sequential_17_dense_42_matmul_readvariableop_resource:АвАE
6sequential_17_dense_42_biasadd_readvariableop_resource:	АI
5sequential_17_dense_43_matmul_readvariableop_resource:
ААE
6sequential_17_dense_43_biasadd_readvariableop_resource:	А:
'dense_44_matmul_readvariableop_resource:	А6
(dense_44_biasadd_readvariableop_resource:
identityИв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpвDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpвFsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в3sequential_16/batch_normalization_16/ReadVariableOpв5sequential_16/batch_normalization_16/ReadVariableOp_1в.sequential_16/conv2d_48/BiasAdd/ReadVariableOpв-sequential_16/conv2d_48/Conv2D/ReadVariableOpв.sequential_16/conv2d_49/BiasAdd/ReadVariableOpв-sequential_16/conv2d_49/Conv2D/ReadVariableOpв.sequential_16/conv2d_50/BiasAdd/ReadVariableOpв-sequential_16/conv2d_50/Conv2D/ReadVariableOpв-sequential_16/dense_40/BiasAdd/ReadVariableOpв,sequential_16/dense_40/MatMul/ReadVariableOpв-sequential_16/dense_41/BiasAdd/ReadVariableOpв,sequential_16/dense_41/MatMul/ReadVariableOpвDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpвFsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в3sequential_17/batch_normalization_17/ReadVariableOpв5sequential_17/batch_normalization_17/ReadVariableOp_1в.sequential_17/conv2d_51/BiasAdd/ReadVariableOpв-sequential_17/conv2d_51/Conv2D/ReadVariableOpв.sequential_17/conv2d_52/BiasAdd/ReadVariableOpв-sequential_17/conv2d_52/Conv2D/ReadVariableOpв.sequential_17/conv2d_53/BiasAdd/ReadVariableOpв-sequential_17/conv2d_53/Conv2D/ReadVariableOpв-sequential_17/dense_42/BiasAdd/ReadVariableOpв,sequential_17/dense_42/MatMul/ReadVariableOpв-sequential_17/dense_43/BiasAdd/ReadVariableOpв,sequential_17/dense_43/MatMul/ReadVariableOp│
+sequential_16/lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_16/lambda_16/strided_slice/stack╖
-sequential_16/lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2/
-sequential_16/lambda_16/strided_slice/stack_1╖
-sequential_16/lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_16/lambda_16/strided_slice/stack_2Ў
%sequential_16/lambda_16/strided_sliceStridedSliceinput_14sequential_16/lambda_16/strided_slice/stack:output:06sequential_16/lambda_16/strided_slice/stack_1:output:06sequential_16/lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_16/lambda_16/strided_sliceу
3sequential_16/batch_normalization_16/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_16/batch_normalization_16/ReadVariableOpщ
5sequential_16/batch_normalization_16/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_16/batch_normalization_16/ReadVariableOp_1Ц
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1╨
5sequential_16/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3.sequential_16/lambda_16/strided_slice:output:0;sequential_16/batch_normalization_16/ReadVariableOp:value:0=sequential_16/batch_normalization_16/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 27
5sequential_16/batch_normalization_16/FusedBatchNormV3▌
-sequential_16/conv2d_48/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_48/Conv2D/ReadVariableOpЮ
sequential_16/conv2d_48/Conv2DConv2D9sequential_16/batch_normalization_16/FusedBatchNormV3:y:05sequential_16/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_16/conv2d_48/Conv2D╘
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpш
sequential_16/conv2d_48/BiasAddBiasAdd'sequential_16/conv2d_48/Conv2D:output:06sequential_16/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_16/conv2d_48/BiasAddи
sequential_16/conv2d_48/ReluRelu(sequential_16/conv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_16/conv2d_48/ReluЇ
&sequential_16/max_pooling2d_48/MaxPoolMaxPool*sequential_16/conv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool▐
-sequential_16/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_16/conv2d_49/Conv2D/ReadVariableOpХ
sequential_16/conv2d_49/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_16/conv2d_49/Conv2D╒
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_49/BiasAddBiasAdd'sequential_16/conv2d_49/Conv2D:output:06sequential_16/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_16/conv2d_49/BiasAddй
sequential_16/conv2d_49/ReluRelu(sequential_16/conv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_16/conv2d_49/Reluї
&sequential_16/max_pooling2d_49/MaxPoolMaxPool*sequential_16/conv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool▀
-sequential_16/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_16/conv2d_50/Conv2D/ReadVariableOpХ
sequential_16/conv2d_50/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:05sequential_16/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_16/conv2d_50/Conv2D╒
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_50/BiasAddBiasAdd'sequential_16/conv2d_50/Conv2D:output:06sequential_16/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_16/conv2d_50/BiasAddй
sequential_16/conv2d_50/ReluRelu(sequential_16/conv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_16/conv2d_50/Reluї
&sequential_16/max_pooling2d_50/MaxPoolMaxPool*sequential_16/conv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPool╛
!sequential_16/dropout_48/IdentityIdentity/sequential_16/max_pooling2d_50/MaxPool:output:0*
T0*0
_output_shapes
:         		А2#
!sequential_16/dropout_48/IdentityС
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_16/flatten_16/Const╪
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_48/Identity:output:0'sequential_16/flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_16/flatten_16/Reshape╒
,sequential_16/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_16/dense_40/MatMul/ReadVariableOp▄
sequential_16/dense_40/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/MatMul╥
-sequential_16/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_40/BiasAdd/ReadVariableOp▐
sequential_16/dense_40/BiasAddBiasAdd'sequential_16/dense_40/MatMul:product:05sequential_16/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_40/BiasAddЮ
sequential_16/dense_40/ReluRelu'sequential_16/dense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/Relu░
!sequential_16/dropout_49/IdentityIdentity)sequential_16/dense_40/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_16/dropout_49/Identity╘
,sequential_16/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_16/dense_41/MatMul/ReadVariableOp▌
sequential_16/dense_41/MatMulMatMul*sequential_16/dropout_49/Identity:output:04sequential_16/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/MatMul╥
-sequential_16/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_41/BiasAdd/ReadVariableOp▐
sequential_16/dense_41/BiasAddBiasAdd'sequential_16/dense_41/MatMul:product:05sequential_16/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_41/BiasAddЮ
sequential_16/dense_41/ReluRelu'sequential_16/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/Relu░
!sequential_16/dropout_50/IdentityIdentity)sequential_16/dense_41/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_16/dropout_50/Identity│
+sequential_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2-
+sequential_17/lambda_17/strided_slice/stack╖
-sequential_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2/
-sequential_17/lambda_17/strided_slice/stack_1╖
-sequential_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_17/lambda_17/strided_slice/stack_2Ў
%sequential_17/lambda_17/strided_sliceStridedSliceinput_14sequential_17/lambda_17/strided_slice/stack:output:06sequential_17/lambda_17/strided_slice/stack_1:output:06sequential_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_17/lambda_17/strided_sliceу
3sequential_17/batch_normalization_17/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_17/batch_normalization_17/ReadVariableOpщ
5sequential_17/batch_normalization_17/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_17/batch_normalization_17/ReadVariableOp_1Ц
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1╨
5sequential_17/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3.sequential_17/lambda_17/strided_slice:output:0;sequential_17/batch_normalization_17/ReadVariableOp:value:0=sequential_17/batch_normalization_17/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 27
5sequential_17/batch_normalization_17/FusedBatchNormV3▌
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOpЮ
sequential_17/conv2d_51/Conv2DConv2D9sequential_17/batch_normalization_17/FusedBatchNormV3:y:05sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_17/conv2d_51/Conv2D╘
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpш
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_17/conv2d_51/BiasAddи
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_17/conv2d_51/ReluЇ
&sequential_17/max_pooling2d_51/MaxPoolMaxPool*sequential_17/conv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_51/MaxPool▐
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOpХ
sequential_17/conv2d_52/Conv2DConv2D/sequential_17/max_pooling2d_51/MaxPool:output:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_17/conv2d_52/Conv2D╒
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_17/conv2d_52/BiasAddй
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_17/conv2d_52/Reluї
&sequential_17/max_pooling2d_52/MaxPoolMaxPool*sequential_17/conv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_52/MaxPool▀
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOpХ
sequential_17/conv2d_53/Conv2DConv2D/sequential_17/max_pooling2d_52/MaxPool:output:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_17/conv2d_53/Conv2D╒
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_17/conv2d_53/BiasAddй
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_17/conv2d_53/Reluї
&sequential_17/max_pooling2d_53/MaxPoolMaxPool*sequential_17/conv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_53/MaxPool╛
!sequential_17/dropout_51/IdentityIdentity/sequential_17/max_pooling2d_53/MaxPool:output:0*
T0*0
_output_shapes
:         		А2#
!sequential_17/dropout_51/IdentityС
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_17/flatten_17/Const╪
 sequential_17/flatten_17/ReshapeReshape*sequential_17/dropout_51/Identity:output:0'sequential_17/flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_17/flatten_17/Reshape╒
,sequential_17/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_17/dense_42/MatMul/ReadVariableOp▄
sequential_17/dense_42/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/MatMul╥
-sequential_17/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_42/BiasAdd/ReadVariableOp▐
sequential_17/dense_42/BiasAddBiasAdd'sequential_17/dense_42/MatMul:product:05sequential_17/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_42/BiasAddЮ
sequential_17/dense_42/ReluRelu'sequential_17/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/Relu░
!sequential_17/dropout_52/IdentityIdentity)sequential_17/dense_42/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_17/dropout_52/Identity╘
,sequential_17/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_17/dense_43/MatMul/ReadVariableOp▌
sequential_17/dense_43/MatMulMatMul*sequential_17/dropout_52/Identity:output:04sequential_17/dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/MatMul╥
-sequential_17/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_43/BiasAdd/ReadVariableOp▐
sequential_17/dense_43/BiasAddBiasAdd'sequential_17/dense_43/MatMul:product:05sequential_17/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_43/BiasAddЮ
sequential_17/dense_43/ReluRelu'sequential_17/dense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/Relu░
!sequential_17/dropout_53/IdentityIdentity)sequential_17/dense_43/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_17/dropout_53/Identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╞
concatConcatV2*sequential_16/dropout_50/Identity:output:0*sequential_17/dropout_53/Identity:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatй
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_44/MatMul/ReadVariableOpЧ
dense_44/MatMulMatMulconcat:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpе
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/BiasAdd|
dense_44/SoftmaxSoftmaxdense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_44/Softmaxч
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul▀
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul▐
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulч
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul▀
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul▐
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulй
IdentityIdentitydense_44/Softmax:softmax:03^conv2d_48/kernel/Regularizer/Square/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOpE^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_16/ReadVariableOp6^sequential_16/batch_normalization_16/ReadVariableOp_1/^sequential_16/conv2d_48/BiasAdd/ReadVariableOp.^sequential_16/conv2d_48/Conv2D/ReadVariableOp/^sequential_16/conv2d_49/BiasAdd/ReadVariableOp.^sequential_16/conv2d_49/Conv2D/ReadVariableOp/^sequential_16/conv2d_50/BiasAdd/ReadVariableOp.^sequential_16/conv2d_50/Conv2D/ReadVariableOp.^sequential_16/dense_40/BiasAdd/ReadVariableOp-^sequential_16/dense_40/MatMul/ReadVariableOp.^sequential_16/dense_41/BiasAdd/ReadVariableOp-^sequential_16/dense_41/MatMul/ReadVariableOpE^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_17/ReadVariableOp6^sequential_17/batch_normalization_17/ReadVariableOp_1/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_42/BiasAdd/ReadVariableOp-^sequential_17/dense_42/MatMul/ReadVariableOp.^sequential_17/dense_43/BiasAdd/ReadVariableOp-^sequential_17/dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2М
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_16/ReadVariableOp3sequential_16/batch_normalization_16/ReadVariableOp2n
5sequential_16/batch_normalization_16/ReadVariableOp_15sequential_16/batch_normalization_16/ReadVariableOp_12`
.sequential_16/conv2d_48/BiasAdd/ReadVariableOp.sequential_16/conv2d_48/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_48/Conv2D/ReadVariableOp-sequential_16/conv2d_48/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_49/BiasAdd/ReadVariableOp.sequential_16/conv2d_49/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_49/Conv2D/ReadVariableOp-sequential_16/conv2d_49/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_50/BiasAdd/ReadVariableOp.sequential_16/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_50/Conv2D/ReadVariableOp-sequential_16/conv2d_50/Conv2D/ReadVariableOp2^
-sequential_16/dense_40/BiasAdd/ReadVariableOp-sequential_16/dense_40/BiasAdd/ReadVariableOp2\
,sequential_16/dense_40/MatMul/ReadVariableOp,sequential_16/dense_40/MatMul/ReadVariableOp2^
-sequential_16/dense_41/BiasAdd/ReadVariableOp-sequential_16/dense_41/BiasAdd/ReadVariableOp2\
,sequential_16/dense_41/MatMul/ReadVariableOp,sequential_16/dense_41/MatMul/ReadVariableOp2М
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_17/ReadVariableOp3sequential_17/batch_normalization_17/ReadVariableOp2n
5sequential_17/batch_normalization_17/ReadVariableOp_15sequential_17/batch_normalization_17/ReadVariableOp_12`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_42/BiasAdd/ReadVariableOp-sequential_17/dense_42/BiasAdd/ReadVariableOp2\
,sequential_17/dense_42/MatMul/ReadVariableOp,sequential_17/dense_42/MatMul/ReadVariableOp2^
-sequential_17/dense_43/BiasAdd/ReadVariableOp-sequential_17/dense_43/BiasAdd/ReadVariableOp2\
,sequential_17/dense_43/MatMul/ReadVariableOp,sequential_17/dense_43/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
й
Ъ
*__inference_dense_43_layer_call_fn_2149028

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_21448322
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
║х
┬
__inference_call_1985057

inputsJ
<sequential_16_batch_normalization_16_readvariableop_resource:L
>sequential_16_batch_normalization_16_readvariableop_1_resource:[
Msequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:]
Osequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_16_conv2d_48_conv2d_readvariableop_resource: E
7sequential_16_conv2d_48_biasadd_readvariableop_resource: Q
6sequential_16_conv2d_49_conv2d_readvariableop_resource: АF
7sequential_16_conv2d_49_biasadd_readvariableop_resource:	АR
6sequential_16_conv2d_50_conv2d_readvariableop_resource:ААF
7sequential_16_conv2d_50_biasadd_readvariableop_resource:	АJ
5sequential_16_dense_40_matmul_readvariableop_resource:АвАE
6sequential_16_dense_40_biasadd_readvariableop_resource:	АI
5sequential_16_dense_41_matmul_readvariableop_resource:
ААE
6sequential_16_dense_41_biasadd_readvariableop_resource:	АJ
<sequential_17_batch_normalization_17_readvariableop_resource:L
>sequential_17_batch_normalization_17_readvariableop_1_resource:[
Msequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_17_conv2d_51_conv2d_readvariableop_resource: E
7sequential_17_conv2d_51_biasadd_readvariableop_resource: Q
6sequential_17_conv2d_52_conv2d_readvariableop_resource: АF
7sequential_17_conv2d_52_biasadd_readvariableop_resource:	АR
6sequential_17_conv2d_53_conv2d_readvariableop_resource:ААF
7sequential_17_conv2d_53_biasadd_readvariableop_resource:	АJ
5sequential_17_dense_42_matmul_readvariableop_resource:АвАE
6sequential_17_dense_42_biasadd_readvariableop_resource:	АI
5sequential_17_dense_43_matmul_readvariableop_resource:
ААE
6sequential_17_dense_43_biasadd_readvariableop_resource:	А:
'dense_44_matmul_readvariableop_resource:	А6
(dense_44_biasadd_readvariableop_resource:
identityИвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpвDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpвFsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в3sequential_16/batch_normalization_16/ReadVariableOpв5sequential_16/batch_normalization_16/ReadVariableOp_1в.sequential_16/conv2d_48/BiasAdd/ReadVariableOpв-sequential_16/conv2d_48/Conv2D/ReadVariableOpв.sequential_16/conv2d_49/BiasAdd/ReadVariableOpв-sequential_16/conv2d_49/Conv2D/ReadVariableOpв.sequential_16/conv2d_50/BiasAdd/ReadVariableOpв-sequential_16/conv2d_50/Conv2D/ReadVariableOpв-sequential_16/dense_40/BiasAdd/ReadVariableOpв,sequential_16/dense_40/MatMul/ReadVariableOpв-sequential_16/dense_41/BiasAdd/ReadVariableOpв,sequential_16/dense_41/MatMul/ReadVariableOpвDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpвFsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в3sequential_17/batch_normalization_17/ReadVariableOpв5sequential_17/batch_normalization_17/ReadVariableOp_1в.sequential_17/conv2d_51/BiasAdd/ReadVariableOpв-sequential_17/conv2d_51/Conv2D/ReadVariableOpв.sequential_17/conv2d_52/BiasAdd/ReadVariableOpв-sequential_17/conv2d_52/Conv2D/ReadVariableOpв.sequential_17/conv2d_53/BiasAdd/ReadVariableOpв-sequential_17/conv2d_53/Conv2D/ReadVariableOpв-sequential_17/dense_42/BiasAdd/ReadVariableOpв,sequential_17/dense_42/MatMul/ReadVariableOpв-sequential_17/dense_43/BiasAdd/ReadVariableOpв,sequential_17/dense_43/MatMul/ReadVariableOp│
+sequential_16/lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_16/lambda_16/strided_slice/stack╖
-sequential_16/lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2/
-sequential_16/lambda_16/strided_slice/stack_1╖
-sequential_16/lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_16/lambda_16/strided_slice/stack_2ї
%sequential_16/lambda_16/strided_sliceStridedSliceinputs4sequential_16/lambda_16/strided_slice/stack:output:06sequential_16/lambda_16/strided_slice/stack_1:output:06sequential_16/lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_16/lambda_16/strided_sliceу
3sequential_16/batch_normalization_16/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_16/batch_normalization_16/ReadVariableOpщ
5sequential_16/batch_normalization_16/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_16/batch_normalization_16/ReadVariableOp_1Ц
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1╨
5sequential_16/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3.sequential_16/lambda_16/strided_slice:output:0;sequential_16/batch_normalization_16/ReadVariableOp:value:0=sequential_16/batch_normalization_16/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 27
5sequential_16/batch_normalization_16/FusedBatchNormV3▌
-sequential_16/conv2d_48/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_48/Conv2D/ReadVariableOpЮ
sequential_16/conv2d_48/Conv2DConv2D9sequential_16/batch_normalization_16/FusedBatchNormV3:y:05sequential_16/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_16/conv2d_48/Conv2D╘
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpш
sequential_16/conv2d_48/BiasAddBiasAdd'sequential_16/conv2d_48/Conv2D:output:06sequential_16/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_16/conv2d_48/BiasAddи
sequential_16/conv2d_48/ReluRelu(sequential_16/conv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_16/conv2d_48/ReluЇ
&sequential_16/max_pooling2d_48/MaxPoolMaxPool*sequential_16/conv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool▐
-sequential_16/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_16/conv2d_49/Conv2D/ReadVariableOpХ
sequential_16/conv2d_49/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_16/conv2d_49/Conv2D╒
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_49/BiasAddBiasAdd'sequential_16/conv2d_49/Conv2D:output:06sequential_16/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_16/conv2d_49/BiasAddй
sequential_16/conv2d_49/ReluRelu(sequential_16/conv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_16/conv2d_49/Reluї
&sequential_16/max_pooling2d_49/MaxPoolMaxPool*sequential_16/conv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool▀
-sequential_16/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_16/conv2d_50/Conv2D/ReadVariableOpХ
sequential_16/conv2d_50/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:05sequential_16/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_16/conv2d_50/Conv2D╒
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_50/BiasAddBiasAdd'sequential_16/conv2d_50/Conv2D:output:06sequential_16/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_16/conv2d_50/BiasAddй
sequential_16/conv2d_50/ReluRelu(sequential_16/conv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_16/conv2d_50/Reluї
&sequential_16/max_pooling2d_50/MaxPoolMaxPool*sequential_16/conv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPool╛
!sequential_16/dropout_48/IdentityIdentity/sequential_16/max_pooling2d_50/MaxPool:output:0*
T0*0
_output_shapes
:         		А2#
!sequential_16/dropout_48/IdentityС
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_16/flatten_16/Const╪
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_48/Identity:output:0'sequential_16/flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_16/flatten_16/Reshape╒
,sequential_16/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_16/dense_40/MatMul/ReadVariableOp▄
sequential_16/dense_40/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/MatMul╥
-sequential_16/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_40/BiasAdd/ReadVariableOp▐
sequential_16/dense_40/BiasAddBiasAdd'sequential_16/dense_40/MatMul:product:05sequential_16/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_40/BiasAddЮ
sequential_16/dense_40/ReluRelu'sequential_16/dense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/Relu░
!sequential_16/dropout_49/IdentityIdentity)sequential_16/dense_40/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_16/dropout_49/Identity╘
,sequential_16/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_16/dense_41/MatMul/ReadVariableOp▌
sequential_16/dense_41/MatMulMatMul*sequential_16/dropout_49/Identity:output:04sequential_16/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/MatMul╥
-sequential_16/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_41/BiasAdd/ReadVariableOp▐
sequential_16/dense_41/BiasAddBiasAdd'sequential_16/dense_41/MatMul:product:05sequential_16/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_41/BiasAddЮ
sequential_16/dense_41/ReluRelu'sequential_16/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/Relu░
!sequential_16/dropout_50/IdentityIdentity)sequential_16/dense_41/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_16/dropout_50/Identity│
+sequential_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2-
+sequential_17/lambda_17/strided_slice/stack╖
-sequential_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2/
-sequential_17/lambda_17/strided_slice/stack_1╖
-sequential_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_17/lambda_17/strided_slice/stack_2ї
%sequential_17/lambda_17/strided_sliceStridedSliceinputs4sequential_17/lambda_17/strided_slice/stack:output:06sequential_17/lambda_17/strided_slice/stack_1:output:06sequential_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_17/lambda_17/strided_sliceу
3sequential_17/batch_normalization_17/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_17/batch_normalization_17/ReadVariableOpщ
5sequential_17/batch_normalization_17/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_17/batch_normalization_17/ReadVariableOp_1Ц
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1╨
5sequential_17/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3.sequential_17/lambda_17/strided_slice:output:0;sequential_17/batch_normalization_17/ReadVariableOp:value:0=sequential_17/batch_normalization_17/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 27
5sequential_17/batch_normalization_17/FusedBatchNormV3▌
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOpЮ
sequential_17/conv2d_51/Conv2DConv2D9sequential_17/batch_normalization_17/FusedBatchNormV3:y:05sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_17/conv2d_51/Conv2D╘
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpш
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_17/conv2d_51/BiasAddи
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_17/conv2d_51/ReluЇ
&sequential_17/max_pooling2d_51/MaxPoolMaxPool*sequential_17/conv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_51/MaxPool▐
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOpХ
sequential_17/conv2d_52/Conv2DConv2D/sequential_17/max_pooling2d_51/MaxPool:output:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_17/conv2d_52/Conv2D╒
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_17/conv2d_52/BiasAddй
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_17/conv2d_52/Reluї
&sequential_17/max_pooling2d_52/MaxPoolMaxPool*sequential_17/conv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_52/MaxPool▀
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOpХ
sequential_17/conv2d_53/Conv2DConv2D/sequential_17/max_pooling2d_52/MaxPool:output:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_17/conv2d_53/Conv2D╒
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_17/conv2d_53/BiasAddй
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_17/conv2d_53/Reluї
&sequential_17/max_pooling2d_53/MaxPoolMaxPool*sequential_17/conv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_53/MaxPool╛
!sequential_17/dropout_51/IdentityIdentity/sequential_17/max_pooling2d_53/MaxPool:output:0*
T0*0
_output_shapes
:         		А2#
!sequential_17/dropout_51/IdentityС
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_17/flatten_17/Const╪
 sequential_17/flatten_17/ReshapeReshape*sequential_17/dropout_51/Identity:output:0'sequential_17/flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_17/flatten_17/Reshape╒
,sequential_17/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_17/dense_42/MatMul/ReadVariableOp▄
sequential_17/dense_42/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/MatMul╥
-sequential_17/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_42/BiasAdd/ReadVariableOp▐
sequential_17/dense_42/BiasAddBiasAdd'sequential_17/dense_42/MatMul:product:05sequential_17/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_42/BiasAddЮ
sequential_17/dense_42/ReluRelu'sequential_17/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/Relu░
!sequential_17/dropout_52/IdentityIdentity)sequential_17/dense_42/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_17/dropout_52/Identity╘
,sequential_17/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_17/dense_43/MatMul/ReadVariableOp▌
sequential_17/dense_43/MatMulMatMul*sequential_17/dropout_52/Identity:output:04sequential_17/dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/MatMul╥
-sequential_17/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_43/BiasAdd/ReadVariableOp▐
sequential_17/dense_43/BiasAddBiasAdd'sequential_17/dense_43/MatMul:product:05sequential_17/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_43/BiasAddЮ
sequential_17/dense_43/ReluRelu'sequential_17/dense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/Relu░
!sequential_17/dropout_53/IdentityIdentity)sequential_17/dense_43/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_17/dropout_53/Identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╞
concatConcatV2*sequential_16/dropout_50/Identity:output:0*sequential_17/dropout_53/Identity:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatй
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_44/MatMul/ReadVariableOpЧ
dense_44/MatMulMatMulconcat:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpе
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/BiasAdd|
dense_44/SoftmaxSoftmaxdense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_44/Softmaxя
IdentityIdentitydense_44/Softmax:softmax:0 ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOpE^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_16/ReadVariableOp6^sequential_16/batch_normalization_16/ReadVariableOp_1/^sequential_16/conv2d_48/BiasAdd/ReadVariableOp.^sequential_16/conv2d_48/Conv2D/ReadVariableOp/^sequential_16/conv2d_49/BiasAdd/ReadVariableOp.^sequential_16/conv2d_49/Conv2D/ReadVariableOp/^sequential_16/conv2d_50/BiasAdd/ReadVariableOp.^sequential_16/conv2d_50/Conv2D/ReadVariableOp.^sequential_16/dense_40/BiasAdd/ReadVariableOp-^sequential_16/dense_40/MatMul/ReadVariableOp.^sequential_16/dense_41/BiasAdd/ReadVariableOp-^sequential_16/dense_41/MatMul/ReadVariableOpE^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_17/ReadVariableOp6^sequential_17/batch_normalization_17/ReadVariableOp_1/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_42/BiasAdd/ReadVariableOp-^sequential_17/dense_42/MatMul/ReadVariableOp.^sequential_17/dense_43/BiasAdd/ReadVariableOp-^sequential_17/dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2М
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_16/ReadVariableOp3sequential_16/batch_normalization_16/ReadVariableOp2n
5sequential_16/batch_normalization_16/ReadVariableOp_15sequential_16/batch_normalization_16/ReadVariableOp_12`
.sequential_16/conv2d_48/BiasAdd/ReadVariableOp.sequential_16/conv2d_48/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_48/Conv2D/ReadVariableOp-sequential_16/conv2d_48/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_49/BiasAdd/ReadVariableOp.sequential_16/conv2d_49/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_49/Conv2D/ReadVariableOp-sequential_16/conv2d_49/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_50/BiasAdd/ReadVariableOp.sequential_16/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_50/Conv2D/ReadVariableOp-sequential_16/conv2d_50/Conv2D/ReadVariableOp2^
-sequential_16/dense_40/BiasAdd/ReadVariableOp-sequential_16/dense_40/BiasAdd/ReadVariableOp2\
,sequential_16/dense_40/MatMul/ReadVariableOp,sequential_16/dense_40/MatMul/ReadVariableOp2^
-sequential_16/dense_41/BiasAdd/ReadVariableOp-sequential_16/dense_41/BiasAdd/ReadVariableOp2\
,sequential_16/dense_41/MatMul/ReadVariableOp,sequential_16/dense_41/MatMul/ReadVariableOp2М
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_17/ReadVariableOp3sequential_17/batch_normalization_17/ReadVariableOp2n
5sequential_17/batch_normalization_17/ReadVariableOp_15sequential_17/batch_normalization_17/ReadVariableOp_12`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_42/BiasAdd/ReadVariableOp-sequential_17/dense_42/BiasAdd/ReadVariableOp2\
,sequential_17/dense_42/MatMul/ReadVariableOp,sequential_17/dense_42/MatMul/ReadVariableOp2^
-sequential_17/dense_43/BiasAdd/ReadVariableOp-sequential_17/dense_43/BiasAdd/ReadVariableOp2\
,sequential_17/dense_43/MatMul/ReadVariableOp,sequential_17/dense_43/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
й
Ъ
*__inference_dense_41_layer_call_fn_2148617

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_21439622
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╠\
Я
J__inference_sequential_16_layer_call_and_return_conditional_losses_2143994

inputs,
batch_normalization_16_2143831:,
batch_normalization_16_2143833:,
batch_normalization_16_2143835:,
batch_normalization_16_2143837:+
conv2d_48_2143858: 
conv2d_48_2143860: ,
conv2d_49_2143876: А 
conv2d_49_2143878:	А-
conv2d_50_2143894:АА 
conv2d_50_2143896:	А%
dense_40_2143933:АвА
dense_40_2143935:	А$
dense_41_2143963:
АА
dense_41_2143965:	А
identityИв.batch_normalization_16/StatefulPartitionedCallв!conv2d_48/StatefulPartitionedCallв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв!conv2d_49/StatefulPartitionedCallв!conv2d_50/StatefulPartitionedCallв dense_40/StatefulPartitionedCallв1dense_40/kernel/Regularizer/Square/ReadVariableOpв dense_41/StatefulPartitionedCallв1dense_41/kernel/Regularizer/Square/ReadVariableOpх
lambda_16/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8В *O
fJRH
F__inference_lambda_16_layer_call_and_return_conditional_losses_21438112
lambda_16/PartitionedCall╩
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall"lambda_16/PartitionedCall:output:0batch_normalization_16_2143831batch_normalization_16_2143833batch_normalization_16_2143835batch_normalization_16_2143837*
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_214383020
.batch_normalization_16/StatefulPartitionedCall┌
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_48_2143858conv2d_48_2143860*
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
GPU2 *0J 8В *O
fJRH
F__inference_conv2d_48_layer_call_and_return_conditional_losses_21438572#
!conv2d_48/StatefulPartitionedCallЮ
 max_pooling2d_48/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_21437662"
 max_pooling2d_48/PartitionedCall═
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_49_2143876conv2d_49_2143878*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_49_layer_call_and_return_conditional_losses_21438752#
!conv2d_49/StatefulPartitionedCallЯ
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_21437782"
 max_pooling2d_49/PartitionedCall═
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_50_2143894conv2d_50_2143896*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_50_layer_call_and_return_conditional_losses_21438932#
!conv2d_50/StatefulPartitionedCallЯ
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_21437902"
 max_pooling2d_50/PartitionedCallМ
dropout_48/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_21439052
dropout_48/PartitionedCall 
flatten_16/PartitionedCallPartitionedCall#dropout_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_16_layer_call_and_return_conditional_losses_21439132
flatten_16/PartitionedCall║
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_40_2143933dense_40_2143935*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_21439322"
 dense_40/StatefulPartitionedCallД
dropout_49/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_21439432
dropout_49/PartitionedCall║
 dense_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0dense_41_2143963dense_41_2143965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_21439622"
 dense_41/StatefulPartitionedCallД
dropout_50/PartitionedCallPartitionedCall)dense_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_21439732
dropout_50/PartitionedCall┬
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_48_2143858*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul║
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_2143933*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╣
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_41_2143963* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul°
IdentityIdentity#dropout_50/PartitionedCall:output:0/^batch_normalization_16/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall3^conv2d_48/kernel/Regularizer/Square/ReadVariableOp"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp!^dense_41/StatefulPartitionedCall2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
─
b
F__inference_lambda_16_layer_call_and_return_conditional_losses_2144210

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Л■
ф:
#__inference__traced_restore_2149684
file_prefix3
 assignvariableop_dense_44_kernel:	А.
 assignvariableop_1_dense_44_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: =
/assignvariableop_7_batch_normalization_16_gamma:<
.assignvariableop_8_batch_normalization_16_beta:C
5assignvariableop_9_batch_normalization_16_moving_mean:H
:assignvariableop_10_batch_normalization_16_moving_variance:>
$assignvariableop_11_conv2d_48_kernel: 0
"assignvariableop_12_conv2d_48_bias: ?
$assignvariableop_13_conv2d_49_kernel: А1
"assignvariableop_14_conv2d_49_bias:	А@
$assignvariableop_15_conv2d_50_kernel:АА1
"assignvariableop_16_conv2d_50_bias:	А8
#assignvariableop_17_dense_40_kernel:АвА0
!assignvariableop_18_dense_40_bias:	А7
#assignvariableop_19_dense_41_kernel:
АА0
!assignvariableop_20_dense_41_bias:	А>
0assignvariableop_21_batch_normalization_17_gamma:=
/assignvariableop_22_batch_normalization_17_beta:D
6assignvariableop_23_batch_normalization_17_moving_mean:H
:assignvariableop_24_batch_normalization_17_moving_variance:>
$assignvariableop_25_conv2d_51_kernel: 0
"assignvariableop_26_conv2d_51_bias: ?
$assignvariableop_27_conv2d_52_kernel: А1
"assignvariableop_28_conv2d_52_bias:	А@
$assignvariableop_29_conv2d_53_kernel:АА1
"assignvariableop_30_conv2d_53_bias:	А8
#assignvariableop_31_dense_42_kernel:АвА0
!assignvariableop_32_dense_42_bias:	А7
#assignvariableop_33_dense_43_kernel:
АА0
!assignvariableop_34_dense_43_bias:	А#
assignvariableop_35_total: #
assignvariableop_36_count: %
assignvariableop_37_total_1: %
assignvariableop_38_count_1: =
*assignvariableop_39_adam_dense_44_kernel_m:	А6
(assignvariableop_40_adam_dense_44_bias_m:E
7assignvariableop_41_adam_batch_normalization_16_gamma_m:D
6assignvariableop_42_adam_batch_normalization_16_beta_m:E
+assignvariableop_43_adam_conv2d_48_kernel_m: 7
)assignvariableop_44_adam_conv2d_48_bias_m: F
+assignvariableop_45_adam_conv2d_49_kernel_m: А8
)assignvariableop_46_adam_conv2d_49_bias_m:	АG
+assignvariableop_47_adam_conv2d_50_kernel_m:АА8
)assignvariableop_48_adam_conv2d_50_bias_m:	А?
*assignvariableop_49_adam_dense_40_kernel_m:АвА7
(assignvariableop_50_adam_dense_40_bias_m:	А>
*assignvariableop_51_adam_dense_41_kernel_m:
АА7
(assignvariableop_52_adam_dense_41_bias_m:	АE
7assignvariableop_53_adam_batch_normalization_17_gamma_m:D
6assignvariableop_54_adam_batch_normalization_17_beta_m:E
+assignvariableop_55_adam_conv2d_51_kernel_m: 7
)assignvariableop_56_adam_conv2d_51_bias_m: F
+assignvariableop_57_adam_conv2d_52_kernel_m: А8
)assignvariableop_58_adam_conv2d_52_bias_m:	АG
+assignvariableop_59_adam_conv2d_53_kernel_m:АА8
)assignvariableop_60_adam_conv2d_53_bias_m:	А?
*assignvariableop_61_adam_dense_42_kernel_m:АвА7
(assignvariableop_62_adam_dense_42_bias_m:	А>
*assignvariableop_63_adam_dense_43_kernel_m:
АА7
(assignvariableop_64_adam_dense_43_bias_m:	А=
*assignvariableop_65_adam_dense_44_kernel_v:	А6
(assignvariableop_66_adam_dense_44_bias_v:E
7assignvariableop_67_adam_batch_normalization_16_gamma_v:D
6assignvariableop_68_adam_batch_normalization_16_beta_v:E
+assignvariableop_69_adam_conv2d_48_kernel_v: 7
)assignvariableop_70_adam_conv2d_48_bias_v: F
+assignvariableop_71_adam_conv2d_49_kernel_v: А8
)assignvariableop_72_adam_conv2d_49_bias_v:	АG
+assignvariableop_73_adam_conv2d_50_kernel_v:АА8
)assignvariableop_74_adam_conv2d_50_bias_v:	А?
*assignvariableop_75_adam_dense_40_kernel_v:АвА7
(assignvariableop_76_adam_dense_40_bias_v:	А>
*assignvariableop_77_adam_dense_41_kernel_v:
АА7
(assignvariableop_78_adam_dense_41_bias_v:	АE
7assignvariableop_79_adam_batch_normalization_17_gamma_v:D
6assignvariableop_80_adam_batch_normalization_17_beta_v:E
+assignvariableop_81_adam_conv2d_51_kernel_v: 7
)assignvariableop_82_adam_conv2d_51_bias_v: F
+assignvariableop_83_adam_conv2d_52_kernel_v: А8
)assignvariableop_84_adam_conv2d_52_bias_v:	АG
+assignvariableop_85_adam_conv2d_53_kernel_v:АА8
)assignvariableop_86_adam_conv2d_53_bias_v:	А?
*assignvariableop_87_adam_dense_42_kernel_v:АвА7
(assignvariableop_88_adam_dense_42_bias_v:	А>
*assignvariableop_89_adam_dense_43_kernel_v:
АА7
(assignvariableop_90_adam_dense_43_bias_v:	А
identity_92ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90в*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:\*
dtype0*о)
valueд)Bб)\B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╔
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:\*
dtype0*═
value├B└\B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices·
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ж
_output_shapesє
Ё::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*j
dtypes`
^2\	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_44_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_44_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2б
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4г
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5в
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6к
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7┤
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_16_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8│
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_16_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9║
AssignVariableOp_9AssignVariableOp5assignvariableop_9_batch_normalization_16_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┬
AssignVariableOp_10AssignVariableOp:assignvariableop_10_batch_normalization_16_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11м
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_48_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12к
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_48_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13м
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_49_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14к
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_49_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15м
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_50_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16к
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_50_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17л
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_40_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18й
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_40_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19л
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_41_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20й
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_41_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╕
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_17_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╖
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_17_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╛
AssignVariableOp_23AssignVariableOp6assignvariableop_23_batch_normalization_17_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24┬
AssignVariableOp_24AssignVariableOp:assignvariableop_24_batch_normalization_17_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25м
AssignVariableOp_25AssignVariableOp$assignvariableop_25_conv2d_51_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26к
AssignVariableOp_26AssignVariableOp"assignvariableop_26_conv2d_51_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27м
AssignVariableOp_27AssignVariableOp$assignvariableop_27_conv2d_52_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28к
AssignVariableOp_28AssignVariableOp"assignvariableop_28_conv2d_52_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29м
AssignVariableOp_29AssignVariableOp$assignvariableop_29_conv2d_53_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30к
AssignVariableOp_30AssignVariableOp"assignvariableop_30_conv2d_53_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31л
AssignVariableOp_31AssignVariableOp#assignvariableop_31_dense_42_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32й
AssignVariableOp_32AssignVariableOp!assignvariableop_32_dense_42_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33л
AssignVariableOp_33AssignVariableOp#assignvariableop_33_dense_43_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34й
AssignVariableOp_34AssignVariableOp!assignvariableop_34_dense_43_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35б
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36б
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37г
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38г
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_44_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_44_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41┐
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_16_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╛
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_16_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43│
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_48_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44▒
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_48_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45│
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_49_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_49_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_50_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_50_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49▓
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_40_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50░
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_40_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51▓
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_41_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52░
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_41_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53┐
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_17_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54╛
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_17_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55│
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_51_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56▒
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_51_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57│
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_52_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58▒
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_52_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59│
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_53_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60▒
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_53_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61▓
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_42_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62░
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_42_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63▓
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_43_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64░
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_43_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65▓
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_44_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66░
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_44_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67┐
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_16_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68╛
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_16_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69│
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_48_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70▒
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_48_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71│
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_49_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72▒
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_49_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73│
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_50_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74▒
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_50_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75▓
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_dense_40_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76░
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_dense_40_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77▓
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_41_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78░
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_41_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79┐
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_17_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80╛
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_17_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81│
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_conv2d_51_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82▒
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_conv2d_51_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83│
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_conv2d_52_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84▒
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_conv2d_52_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85│
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv2d_53_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86▒
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv2d_53_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87▓
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_42_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88░
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_42_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89▓
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_dense_43_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90░
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_dense_43_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_909
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp░
Identity_91Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_91г
Identity_92IdentityIdentity_91:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90*
T0*
_output_shapes
: 2
Identity_92"#
identity_92Identity_92:output:0*═
_input_shapes╗
╕: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_90AssignVariableOp_90:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Н
Ю
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148379

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
р
N
2__inference_max_pooling2d_52_layer_call_fn_2144654

inputs
identityє
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_21446482
PartitionedCallП
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
║
н
E__inference_dense_41_layer_call_and_return_conditional_losses_2143962

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╟
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─
b
F__inference_lambda_16_layer_call_and_return_conditional_losses_2148301

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╫
e
,__inference_dropout_52_layer_call_fn_2148996

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_21449482
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┴
┬
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2144570

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
ь
╡
__inference_loss_fn_4_2149094O
:dense_42_kernel_regularizer_square_readvariableop_resource:АвА
identityИв1dense_42/kernel/Regularizer/Square/ReadVariableOpф
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_42_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mulЪ
IdentityIdentity#dense_42/kernel/Regularizer/mul:z:02^dense_42/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp
╦
H
,__inference_dropout_52_layer_call_fn_2148991

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_21448132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╫
e
,__inference_dropout_50_layer_call_fn_2148644

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_21440452
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─
b
F__inference_lambda_16_layer_call_and_return_conditional_losses_2148309

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
й
╫
*__inference_CNN_2jet_layer_call_fn_2146317

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: %

unknown_19: А

unknown_20:	А&

unknown_21:АА

unknown_22:	А

unknown_23:АвА

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCall∙
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
:         *@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_21454952
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ы
H
,__inference_dropout_48_layer_call_fn_2148510

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_21439052
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
░a
О	
J__inference_sequential_17_layer_call_and_return_conditional_losses_2145182

inputs,
batch_normalization_17_2145122:,
batch_normalization_17_2145124:,
batch_normalization_17_2145126:,
batch_normalization_17_2145128:+
conv2d_51_2145131: 
conv2d_51_2145133: ,
conv2d_52_2145137: А 
conv2d_52_2145139:	А-
conv2d_53_2145143:АА 
conv2d_53_2145145:	А%
dense_42_2145151:АвА
dense_42_2145153:	А$
dense_43_2145157:
АА
dense_43_2145159:	А
identityИв.batch_normalization_17/StatefulPartitionedCallв!conv2d_51/StatefulPartitionedCallв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв!conv2d_52/StatefulPartitionedCallв!conv2d_53/StatefulPartitionedCallв dense_42/StatefulPartitionedCallв1dense_42/kernel/Regularizer/Square/ReadVariableOpв dense_43/StatefulPartitionedCallв1dense_43/kernel/Regularizer/Square/ReadVariableOpв"dropout_51/StatefulPartitionedCallв"dropout_52/StatefulPartitionedCallв"dropout_53/StatefulPartitionedCallх
lambda_17/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8В *O
fJRH
F__inference_lambda_17_layer_call_and_return_conditional_losses_21450802
lambda_17/PartitionedCall╚
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall"lambda_17/PartitionedCall:output:0batch_normalization_17_2145122batch_normalization_17_2145124batch_normalization_17_2145126batch_normalization_17_2145128*
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_214505320
.batch_normalization_17/StatefulPartitionedCall┌
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0conv2d_51_2145131conv2d_51_2145133*
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
GPU2 *0J 8В *O
fJRH
F__inference_conv2d_51_layer_call_and_return_conditional_losses_21447272#
!conv2d_51/StatefulPartitionedCallЮ
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_21446362"
 max_pooling2d_51/PartitionedCall═
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0conv2d_52_2145137conv2d_52_2145139*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_52_layer_call_and_return_conditional_losses_21447452#
!conv2d_52/StatefulPartitionedCallЯ
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_21446482"
 max_pooling2d_52/PartitionedCall═
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0conv2d_53_2145143conv2d_53_2145145*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_53_layer_call_and_return_conditional_losses_21447632#
!conv2d_53/StatefulPartitionedCallЯ
 max_pooling2d_53/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_21446602"
 max_pooling2d_53/PartitionedCallд
"dropout_51/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_53/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_51_layer_call_and_return_conditional_losses_21449872$
"dropout_51/StatefulPartitionedCallЗ
flatten_17/PartitionedCallPartitionedCall+dropout_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_17_layer_call_and_return_conditional_losses_21447832
flatten_17/PartitionedCall║
 dense_42/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_42_2145151dense_42_2145153*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_21448022"
 dense_42/StatefulPartitionedCall┴
"dropout_52/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0#^dropout_51/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_21449482$
"dropout_52/StatefulPartitionedCall┬
 dense_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_52/StatefulPartitionedCall:output:0dense_43_2145157dense_43_2145159*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_21448322"
 dense_43/StatefulPartitionedCall┴
"dropout_53/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0#^dropout_52/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_21449152$
"dropout_53/StatefulPartitionedCall┬
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_51_2145131*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul║
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_42_2145151*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╣
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_43_2145157* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulя
IdentityIdentity+dropout_53/StatefulPartitionedCall:output:0/^batch_normalization_17/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall2^dense_42/kernel/Regularizer/Square/ReadVariableOp!^dense_43/StatefulPartitionedCall2^dense_43/kernel/Regularizer/Square/ReadVariableOp#^dropout_51/StatefulPartitionedCall#^dropout_52/StatefulPartitionedCall#^dropout_53/StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_51/StatefulPartitionedCall"dropout_51/StatefulPartitionedCall2H
"dropout_52/StatefulPartitionedCall"dropout_52/StatefulPartitionedCall2H
"dropout_53/StatefulPartitionedCall"dropout_53/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Вт
┬
__inference_call_1988836

inputsJ
<sequential_16_batch_normalization_16_readvariableop_resource:L
>sequential_16_batch_normalization_16_readvariableop_1_resource:[
Msequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:]
Osequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_16_conv2d_48_conv2d_readvariableop_resource: E
7sequential_16_conv2d_48_biasadd_readvariableop_resource: Q
6sequential_16_conv2d_49_conv2d_readvariableop_resource: АF
7sequential_16_conv2d_49_biasadd_readvariableop_resource:	АR
6sequential_16_conv2d_50_conv2d_readvariableop_resource:ААF
7sequential_16_conv2d_50_biasadd_readvariableop_resource:	АJ
5sequential_16_dense_40_matmul_readvariableop_resource:АвАE
6sequential_16_dense_40_biasadd_readvariableop_resource:	АI
5sequential_16_dense_41_matmul_readvariableop_resource:
ААE
6sequential_16_dense_41_biasadd_readvariableop_resource:	АJ
<sequential_17_batch_normalization_17_readvariableop_resource:L
>sequential_17_batch_normalization_17_readvariableop_1_resource:[
Msequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_17_conv2d_51_conv2d_readvariableop_resource: E
7sequential_17_conv2d_51_biasadd_readvariableop_resource: Q
6sequential_17_conv2d_52_conv2d_readvariableop_resource: АF
7sequential_17_conv2d_52_biasadd_readvariableop_resource:	АR
6sequential_17_conv2d_53_conv2d_readvariableop_resource:ААF
7sequential_17_conv2d_53_biasadd_readvariableop_resource:	АJ
5sequential_17_dense_42_matmul_readvariableop_resource:АвАE
6sequential_17_dense_42_biasadd_readvariableop_resource:	АI
5sequential_17_dense_43_matmul_readvariableop_resource:
ААE
6sequential_17_dense_43_biasadd_readvariableop_resource:	А:
'dense_44_matmul_readvariableop_resource:	А6
(dense_44_biasadd_readvariableop_resource:
identityИвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpвDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpвFsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в3sequential_16/batch_normalization_16/ReadVariableOpв5sequential_16/batch_normalization_16/ReadVariableOp_1в.sequential_16/conv2d_48/BiasAdd/ReadVariableOpв-sequential_16/conv2d_48/Conv2D/ReadVariableOpв.sequential_16/conv2d_49/BiasAdd/ReadVariableOpв-sequential_16/conv2d_49/Conv2D/ReadVariableOpв.sequential_16/conv2d_50/BiasAdd/ReadVariableOpв-sequential_16/conv2d_50/Conv2D/ReadVariableOpв-sequential_16/dense_40/BiasAdd/ReadVariableOpв,sequential_16/dense_40/MatMul/ReadVariableOpв-sequential_16/dense_41/BiasAdd/ReadVariableOpв,sequential_16/dense_41/MatMul/ReadVariableOpвDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpвFsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в3sequential_17/batch_normalization_17/ReadVariableOpв5sequential_17/batch_normalization_17/ReadVariableOp_1в.sequential_17/conv2d_51/BiasAdd/ReadVariableOpв-sequential_17/conv2d_51/Conv2D/ReadVariableOpв.sequential_17/conv2d_52/BiasAdd/ReadVariableOpв-sequential_17/conv2d_52/Conv2D/ReadVariableOpв.sequential_17/conv2d_53/BiasAdd/ReadVariableOpв-sequential_17/conv2d_53/Conv2D/ReadVariableOpв-sequential_17/dense_42/BiasAdd/ReadVariableOpв,sequential_17/dense_42/MatMul/ReadVariableOpв-sequential_17/dense_43/BiasAdd/ReadVariableOpв,sequential_17/dense_43/MatMul/ReadVariableOp│
+sequential_16/lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_16/lambda_16/strided_slice/stack╖
-sequential_16/lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2/
-sequential_16/lambda_16/strided_slice/stack_1╖
-sequential_16/lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_16/lambda_16/strided_slice/stack_2э
%sequential_16/lambda_16/strided_sliceStridedSliceinputs4sequential_16/lambda_16/strided_slice/stack:output:06sequential_16/lambda_16/strided_slice/stack_1:output:06sequential_16/lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2'
%sequential_16/lambda_16/strided_sliceу
3sequential_16/batch_normalization_16/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_16/batch_normalization_16/ReadVariableOpщ
5sequential_16/batch_normalization_16/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_16/batch_normalization_16/ReadVariableOp_1Ц
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1╚
5sequential_16/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3.sequential_16/lambda_16/strided_slice:output:0;sequential_16/batch_normalization_16/ReadVariableOp:value:0=sequential_16/batch_normalization_16/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 27
5sequential_16/batch_normalization_16/FusedBatchNormV3▌
-sequential_16/conv2d_48/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_48/Conv2D/ReadVariableOpЦ
sequential_16/conv2d_48/Conv2DConv2D9sequential_16/batch_normalization_16/FusedBatchNormV3:y:05sequential_16/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2 
sequential_16/conv2d_48/Conv2D╘
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpр
sequential_16/conv2d_48/BiasAddBiasAdd'sequential_16/conv2d_48/Conv2D:output:06sequential_16/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2!
sequential_16/conv2d_48/BiasAddа
sequential_16/conv2d_48/ReluRelu(sequential_16/conv2d_48/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_16/conv2d_48/Reluь
&sequential_16/max_pooling2d_48/MaxPoolMaxPool*sequential_16/conv2d_48/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool▐
-sequential_16/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_16/conv2d_49/Conv2D/ReadVariableOpН
sequential_16/conv2d_49/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2 
sequential_16/conv2d_49/Conv2D╒
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpс
sequential_16/conv2d_49/BiasAddBiasAdd'sequential_16/conv2d_49/Conv2D:output:06sequential_16/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2!
sequential_16/conv2d_49/BiasAddб
sequential_16/conv2d_49/ReluRelu(sequential_16/conv2d_49/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_16/conv2d_49/Reluэ
&sequential_16/max_pooling2d_49/MaxPoolMaxPool*sequential_16/conv2d_49/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool▀
-sequential_16/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_16/conv2d_50/Conv2D/ReadVariableOpН
sequential_16/conv2d_50/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:05sequential_16/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2 
sequential_16/conv2d_50/Conv2D╒
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpс
sequential_16/conv2d_50/BiasAddBiasAdd'sequential_16/conv2d_50/Conv2D:output:06sequential_16/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2!
sequential_16/conv2d_50/BiasAddб
sequential_16/conv2d_50/ReluRelu(sequential_16/conv2d_50/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_16/conv2d_50/Reluэ
&sequential_16/max_pooling2d_50/MaxPoolMaxPool*sequential_16/conv2d_50/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPool╢
!sequential_16/dropout_48/IdentityIdentity/sequential_16/max_pooling2d_50/MaxPool:output:0*
T0*(
_output_shapes
:А		А2#
!sequential_16/dropout_48/IdentityС
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_16/flatten_16/Const╨
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_48/Identity:output:0'sequential_16/flatten_16/Const:output:0*
T0*!
_output_shapes
:ААв2"
 sequential_16/flatten_16/Reshape╒
,sequential_16/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_16/dense_40/MatMul/ReadVariableOp╘
sequential_16/dense_40/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_40/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_16/dense_40/MatMul╥
-sequential_16/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_40/BiasAdd/ReadVariableOp╓
sequential_16/dense_40/BiasAddBiasAdd'sequential_16/dense_40/MatMul:product:05sequential_16/dense_40/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2 
sequential_16/dense_40/BiasAddЦ
sequential_16/dense_40/ReluRelu'sequential_16/dense_40/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_16/dense_40/Reluи
!sequential_16/dropout_49/IdentityIdentity)sequential_16/dense_40/Relu:activations:0*
T0* 
_output_shapes
:
АА2#
!sequential_16/dropout_49/Identity╘
,sequential_16/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_16/dense_41/MatMul/ReadVariableOp╒
sequential_16/dense_41/MatMulMatMul*sequential_16/dropout_49/Identity:output:04sequential_16/dense_41/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_16/dense_41/MatMul╥
-sequential_16/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_41/BiasAdd/ReadVariableOp╓
sequential_16/dense_41/BiasAddBiasAdd'sequential_16/dense_41/MatMul:product:05sequential_16/dense_41/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2 
sequential_16/dense_41/BiasAddЦ
sequential_16/dense_41/ReluRelu'sequential_16/dense_41/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_16/dense_41/Reluи
!sequential_16/dropout_50/IdentityIdentity)sequential_16/dense_41/Relu:activations:0*
T0* 
_output_shapes
:
АА2#
!sequential_16/dropout_50/Identity│
+sequential_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2-
+sequential_17/lambda_17/strided_slice/stack╖
-sequential_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2/
-sequential_17/lambda_17/strided_slice/stack_1╖
-sequential_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_17/lambda_17/strided_slice/stack_2э
%sequential_17/lambda_17/strided_sliceStridedSliceinputs4sequential_17/lambda_17/strided_slice/stack:output:06sequential_17/lambda_17/strided_slice/stack_1:output:06sequential_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2'
%sequential_17/lambda_17/strided_sliceу
3sequential_17/batch_normalization_17/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_17/batch_normalization_17/ReadVariableOpщ
5sequential_17/batch_normalization_17/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_17/batch_normalization_17/ReadVariableOp_1Ц
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1╚
5sequential_17/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3.sequential_17/lambda_17/strided_slice:output:0;sequential_17/batch_normalization_17/ReadVariableOp:value:0=sequential_17/batch_normalization_17/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 27
5sequential_17/batch_normalization_17/FusedBatchNormV3▌
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOpЦ
sequential_17/conv2d_51/Conv2DConv2D9sequential_17/batch_normalization_17/FusedBatchNormV3:y:05sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2 
sequential_17/conv2d_51/Conv2D╘
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpр
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2!
sequential_17/conv2d_51/BiasAddа
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_17/conv2d_51/Reluь
&sequential_17/max_pooling2d_51/MaxPoolMaxPool*sequential_17/conv2d_51/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_51/MaxPool▐
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOpН
sequential_17/conv2d_52/Conv2DConv2D/sequential_17/max_pooling2d_51/MaxPool:output:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2 
sequential_17/conv2d_52/Conv2D╒
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpс
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2!
sequential_17/conv2d_52/BiasAddб
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_17/conv2d_52/Reluэ
&sequential_17/max_pooling2d_52/MaxPoolMaxPool*sequential_17/conv2d_52/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_52/MaxPool▀
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOpН
sequential_17/conv2d_53/Conv2DConv2D/sequential_17/max_pooling2d_52/MaxPool:output:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2 
sequential_17/conv2d_53/Conv2D╒
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpс
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2!
sequential_17/conv2d_53/BiasAddб
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_17/conv2d_53/Reluэ
&sequential_17/max_pooling2d_53/MaxPoolMaxPool*sequential_17/conv2d_53/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_53/MaxPool╢
!sequential_17/dropout_51/IdentityIdentity/sequential_17/max_pooling2d_53/MaxPool:output:0*
T0*(
_output_shapes
:А		А2#
!sequential_17/dropout_51/IdentityС
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_17/flatten_17/Const╨
 sequential_17/flatten_17/ReshapeReshape*sequential_17/dropout_51/Identity:output:0'sequential_17/flatten_17/Const:output:0*
T0*!
_output_shapes
:ААв2"
 sequential_17/flatten_17/Reshape╒
,sequential_17/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_17/dense_42/MatMul/ReadVariableOp╘
sequential_17/dense_42/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_42/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_17/dense_42/MatMul╥
-sequential_17/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_42/BiasAdd/ReadVariableOp╓
sequential_17/dense_42/BiasAddBiasAdd'sequential_17/dense_42/MatMul:product:05sequential_17/dense_42/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2 
sequential_17/dense_42/BiasAddЦ
sequential_17/dense_42/ReluRelu'sequential_17/dense_42/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_17/dense_42/Reluи
!sequential_17/dropout_52/IdentityIdentity)sequential_17/dense_42/Relu:activations:0*
T0* 
_output_shapes
:
АА2#
!sequential_17/dropout_52/Identity╘
,sequential_17/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_17/dense_43/MatMul/ReadVariableOp╒
sequential_17/dense_43/MatMulMatMul*sequential_17/dropout_52/Identity:output:04sequential_17/dense_43/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_17/dense_43/MatMul╥
-sequential_17/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_43/BiasAdd/ReadVariableOp╓
sequential_17/dense_43/BiasAddBiasAdd'sequential_17/dense_43/MatMul:product:05sequential_17/dense_43/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2 
sequential_17/dense_43/BiasAddЦ
sequential_17/dense_43/ReluRelu'sequential_17/dense_43/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_17/dense_43/Reluи
!sequential_17/dropout_53/IdentityIdentity)sequential_17/dense_43/Relu:activations:0*
T0* 
_output_shapes
:
АА2#
!sequential_17/dropout_53/Identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╛
concatConcatV2*sequential_16/dropout_50/Identity:output:0*sequential_17/dropout_53/Identity:output:0concat/axis:output:0*
N*
T0* 
_output_shapes
:
АА2
concatй
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_44/MatMul/ReadVariableOpП
dense_44/MatMulMatMulconcat:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpЭ
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_44/BiasAddt
dense_44/SoftmaxSoftmaxdense_44/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_44/Softmaxч
IdentityIdentitydense_44/Softmax:softmax:0 ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOpE^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_16/ReadVariableOp6^sequential_16/batch_normalization_16/ReadVariableOp_1/^sequential_16/conv2d_48/BiasAdd/ReadVariableOp.^sequential_16/conv2d_48/Conv2D/ReadVariableOp/^sequential_16/conv2d_49/BiasAdd/ReadVariableOp.^sequential_16/conv2d_49/Conv2D/ReadVariableOp/^sequential_16/conv2d_50/BiasAdd/ReadVariableOp.^sequential_16/conv2d_50/Conv2D/ReadVariableOp.^sequential_16/dense_40/BiasAdd/ReadVariableOp-^sequential_16/dense_40/MatMul/ReadVariableOp.^sequential_16/dense_41/BiasAdd/ReadVariableOp-^sequential_16/dense_41/MatMul/ReadVariableOpE^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_17/ReadVariableOp6^sequential_17/batch_normalization_17/ReadVariableOp_1/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_42/BiasAdd/ReadVariableOp-^sequential_17/dense_42/MatMul/ReadVariableOp.^sequential_17/dense_43/BiasAdd/ReadVariableOp-^sequential_17/dense_43/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:АKK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2М
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_16/ReadVariableOp3sequential_16/batch_normalization_16/ReadVariableOp2n
5sequential_16/batch_normalization_16/ReadVariableOp_15sequential_16/batch_normalization_16/ReadVariableOp_12`
.sequential_16/conv2d_48/BiasAdd/ReadVariableOp.sequential_16/conv2d_48/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_48/Conv2D/ReadVariableOp-sequential_16/conv2d_48/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_49/BiasAdd/ReadVariableOp.sequential_16/conv2d_49/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_49/Conv2D/ReadVariableOp-sequential_16/conv2d_49/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_50/BiasAdd/ReadVariableOp.sequential_16/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_50/Conv2D/ReadVariableOp-sequential_16/conv2d_50/Conv2D/ReadVariableOp2^
-sequential_16/dense_40/BiasAdd/ReadVariableOp-sequential_16/dense_40/BiasAdd/ReadVariableOp2\
,sequential_16/dense_40/MatMul/ReadVariableOp,sequential_16/dense_40/MatMul/ReadVariableOp2^
-sequential_16/dense_41/BiasAdd/ReadVariableOp-sequential_16/dense_41/BiasAdd/ReadVariableOp2\
,sequential_16/dense_41/MatMul/ReadVariableOp,sequential_16/dense_41/MatMul/ReadVariableOp2М
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_17/ReadVariableOp3sequential_17/batch_normalization_17/ReadVariableOp2n
5sequential_17/batch_normalization_17/ReadVariableOp_15sequential_17/batch_normalization_17/ReadVariableOp_12`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_42/BiasAdd/ReadVariableOp-sequential_17/dense_42/BiasAdd/ReadVariableOp2\
,sequential_17/dense_42/MatMul/ReadVariableOp,sequential_17/dense_42/MatMul/ReadVariableOp2^
-sequential_17/dense_43/BiasAdd/ReadVariableOp-sequential_17/dense_43/BiasAdd/ReadVariableOp2\
,sequential_17/dense_43/MatMul/ReadVariableOp,sequential_17/dense_43/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
╫
e
,__inference_dropout_53_layer_call_fn_2149055

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_21449152
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
°
e
G__inference_dropout_53_layer_call_and_return_conditional_losses_2144843

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ё
╙
8__inference_batch_normalization_16_layer_call_fn_2148335

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall║
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_21437002
StatefulPartitionedCallи
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
║
н
E__inference_dense_43_layer_call_and_return_conditional_losses_2149045

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╟
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
е
Ш
*__inference_dense_44_layer_call_fn_2148272

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall·
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
GPU2 *0J 8В *N
fIRG
E__inference_dense_44_layer_call_and_return_conditional_losses_21454522
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╣

ў
E__inference_dense_44_layer_call_and_return_conditional_losses_2148283

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╫
e
,__inference_dropout_49_layer_call_fn_2148585

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_21440782
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ў
e
,__inference_dropout_48_layer_call_fn_2148515

inputs
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_21441172
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
ЖЫ
·
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147552

inputs<
.batch_normalization_16_readvariableop_resource:>
0batch_normalization_16_readvariableop_1_resource:M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_48_conv2d_readvariableop_resource: 7
)conv2d_48_biasadd_readvariableop_resource: C
(conv2d_49_conv2d_readvariableop_resource: А8
)conv2d_49_biasadd_readvariableop_resource:	АD
(conv2d_50_conv2d_readvariableop_resource:АА8
)conv2d_50_biasadd_readvariableop_resource:	А<
'dense_40_matmul_readvariableop_resource:АвА7
(dense_40_biasadd_readvariableop_resource:	А;
'dense_41_matmul_readvariableop_resource:
АА7
(dense_41_biasadd_readvariableop_resource:	А
identityИв%batch_normalization_16/AssignNewValueв'batch_normalization_16/AssignNewValue_1в6batch_normalization_16/FusedBatchNormV3/ReadVariableOpв8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_16/ReadVariableOpв'batch_normalization_16/ReadVariableOp_1в conv2d_48/BiasAdd/ReadVariableOpвconv2d_48/Conv2D/ReadVariableOpв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв conv2d_49/BiasAdd/ReadVariableOpвconv2d_49/Conv2D/ReadVariableOpв conv2d_50/BiasAdd/ReadVariableOpвconv2d_50/Conv2D/ReadVariableOpвdense_40/BiasAdd/ReadVariableOpвdense_40/MatMul/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpвdense_41/BiasAdd/ReadVariableOpвdense_41/MatMul/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpЧ
lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_16/strided_slice/stackЫ
lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2!
lambda_16/strided_slice/stack_1Ы
lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2!
lambda_16/strided_slice/stack_2п
lambda_16/strided_sliceStridedSliceinputs&lambda_16/strided_slice/stack:output:0(lambda_16/strided_slice/stack_1:output:0(lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_16/strided_slice╣
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp┐
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1ь
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1№
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 lambda_16/strided_slice:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_16/FusedBatchNormV3╡
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValue┴
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1│
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_48/Conv2D/ReadVariableOpц
conv2d_48/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_48/Conv2Dк
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp░
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_48/BiasAdd~
conv2d_48/ReluReluconv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_48/Relu╩
max_pooling2d_48/MaxPoolMaxPoolconv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_48/MaxPool┤
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_49/Conv2D/ReadVariableOp▌
conv2d_49/Conv2DConv2D!max_pooling2d_48/MaxPool:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_49/Conv2Dл
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp▒
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_49/BiasAdd
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_49/Relu╦
max_pooling2d_49/MaxPoolMaxPoolconv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool╡
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_50/Conv2D/ReadVariableOp▌
conv2d_50/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_50/Conv2Dл
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp▒
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_50/BiasAdd
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_50/Relu╦
max_pooling2d_50/MaxPoolMaxPoolconv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPooly
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_48/dropout/Const╕
dropout_48/dropout/MulMul!max_pooling2d_50/MaxPool:output:0!dropout_48/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout_48/dropout/MulЕ
dropout_48/dropout/ShapeShape!max_pooling2d_50/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_48/dropout/Shape▐
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype021
/dropout_48/dropout/random_uniform/RandomUniformЛ
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_48/dropout/GreaterEqual/yє
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2!
dropout_48/dropout/GreaterEqualй
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout_48/dropout/Castп
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout_48/dropout/Mul_1u
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_16/Constа
flatten_16/ReshapeReshapedropout_48/dropout/Mul_1:z:0flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_16/Reshapeл
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02 
dense_40/MatMul/ReadVariableOpд
dense_40/MatMulMatMulflatten_16/Reshape:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_40/MatMulи
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_40/BiasAdd/ReadVariableOpж
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_40/BiasAddt
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_40/Reluy
dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_49/dropout/Constк
dropout_49/dropout/MulMuldense_40/Relu:activations:0!dropout_49/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_49/dropout/Mul
dropout_49/dropout/ShapeShapedense_40/Relu:activations:0*
T0*
_output_shapes
:2
dropout_49/dropout/Shape╓
/dropout_49/dropout/random_uniform/RandomUniformRandomUniform!dropout_49/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_49/dropout/random_uniform/RandomUniformЛ
!dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_49/dropout/GreaterEqual/yы
dropout_49/dropout/GreaterEqualGreaterEqual8dropout_49/dropout/random_uniform/RandomUniform:output:0*dropout_49/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_49/dropout/GreaterEqualб
dropout_49/dropout/CastCast#dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_49/dropout/Castз
dropout_49/dropout/Mul_1Muldropout_49/dropout/Mul:z:0dropout_49/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_49/dropout/Mul_1к
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_41/MatMul/ReadVariableOpе
dense_41/MatMulMatMuldropout_49/dropout/Mul_1:z:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_41/MatMulи
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_41/BiasAdd/ReadVariableOpж
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_41/BiasAddt
dense_41/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_41/Reluy
dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_50/dropout/Constк
dropout_50/dropout/MulMuldense_41/Relu:activations:0!dropout_50/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_50/dropout/Mul
dropout_50/dropout/ShapeShapedense_41/Relu:activations:0*
T0*
_output_shapes
:2
dropout_50/dropout/Shape╓
/dropout_50/dropout/random_uniform/RandomUniformRandomUniform!dropout_50/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_50/dropout/random_uniform/RandomUniformЛ
!dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_50/dropout/GreaterEqual/yы
dropout_50/dropout/GreaterEqualGreaterEqual8dropout_50/dropout/random_uniform/RandomUniform:output:0*dropout_50/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_50/dropout/GreaterEqualб
dropout_50/dropout/CastCast#dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_50/dropout/Castз
dropout_50/dropout/Mul_1Muldropout_50/dropout/Mul:z:0dropout_50/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_50/dropout/Mul_1┘
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul╤
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╨
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul√
IdentityIdentitydropout_50/dropout/Mul_1:z:0&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp3^conv2d_48/kernel/Regularizer/Square/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
║
н
E__inference_dense_43_layer_call_and_return_conditional_losses_2144832

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╟
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─v
и
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147448

inputs<
.batch_normalization_16_readvariableop_resource:>
0batch_normalization_16_readvariableop_1_resource:M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_48_conv2d_readvariableop_resource: 7
)conv2d_48_biasadd_readvariableop_resource: C
(conv2d_49_conv2d_readvariableop_resource: А8
)conv2d_49_biasadd_readvariableop_resource:	АD
(conv2d_50_conv2d_readvariableop_resource:АА8
)conv2d_50_biasadd_readvariableop_resource:	А<
'dense_40_matmul_readvariableop_resource:АвА7
(dense_40_biasadd_readvariableop_resource:	А;
'dense_41_matmul_readvariableop_resource:
АА7
(dense_41_biasadd_readvariableop_resource:	А
identityИв6batch_normalization_16/FusedBatchNormV3/ReadVariableOpв8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_16/ReadVariableOpв'batch_normalization_16/ReadVariableOp_1в conv2d_48/BiasAdd/ReadVariableOpвconv2d_48/Conv2D/ReadVariableOpв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв conv2d_49/BiasAdd/ReadVariableOpвconv2d_49/Conv2D/ReadVariableOpв conv2d_50/BiasAdd/ReadVariableOpвconv2d_50/Conv2D/ReadVariableOpвdense_40/BiasAdd/ReadVariableOpвdense_40/MatMul/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpвdense_41/BiasAdd/ReadVariableOpвdense_41/MatMul/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpЧ
lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_16/strided_slice/stackЫ
lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2!
lambda_16/strided_slice/stack_1Ы
lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2!
lambda_16/strided_slice/stack_2п
lambda_16/strided_sliceStridedSliceinputs&lambda_16/strided_slice/stack:output:0(lambda_16/strided_slice/stack_1:output:0(lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_16/strided_slice╣
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp┐
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1ь
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ю
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 lambda_16/strided_slice:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3│
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_48/Conv2D/ReadVariableOpц
conv2d_48/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_48/Conv2Dк
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp░
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_48/BiasAdd~
conv2d_48/ReluReluconv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_48/Relu╩
max_pooling2d_48/MaxPoolMaxPoolconv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_48/MaxPool┤
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_49/Conv2D/ReadVariableOp▌
conv2d_49/Conv2DConv2D!max_pooling2d_48/MaxPool:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_49/Conv2Dл
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp▒
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_49/BiasAdd
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_49/Relu╦
max_pooling2d_49/MaxPoolMaxPoolconv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool╡
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_50/Conv2D/ReadVariableOp▌
conv2d_50/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_50/Conv2Dл
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp▒
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_50/BiasAdd
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_50/Relu╦
max_pooling2d_50/MaxPoolMaxPoolconv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPoolФ
dropout_48/IdentityIdentity!max_pooling2d_50/MaxPool:output:0*
T0*0
_output_shapes
:         		А2
dropout_48/Identityu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_16/Constа
flatten_16/ReshapeReshapedropout_48/Identity:output:0flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_16/Reshapeл
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02 
dense_40/MatMul/ReadVariableOpд
dense_40/MatMulMatMulflatten_16/Reshape:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_40/MatMulи
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_40/BiasAdd/ReadVariableOpж
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_40/BiasAddt
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_40/ReluЖ
dropout_49/IdentityIdentitydense_40/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_49/Identityк
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_41/MatMul/ReadVariableOpе
dense_41/MatMulMatMuldropout_49/Identity:output:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_41/MatMulи
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_41/BiasAdd/ReadVariableOpж
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_41/BiasAddt
dense_41/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_41/ReluЖ
dropout_50/IdentityIdentitydense_41/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_50/Identity┘
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul╤
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╨
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulй
IdentityIdentitydropout_50/Identity:output:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp3^conv2d_48/kernel/Regularizer/Square/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╪
 
/__inference_sequential_16_layer_call_fn_2147299

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallЯ
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
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_21439942
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
∙
┬
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2144183

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
х
G
+__inference_lambda_16_layer_call_fn_2148288

inputs
identity╤
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
GPU2 *0J 8В *O
fJRH
F__inference_lambda_16_layer_call_and_return_conditional_losses_21438112
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▀v
▒
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148159
lambda_17_input<
.batch_normalization_17_readvariableop_resource:>
0batch_normalization_17_readvariableop_1_resource:M
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_51_conv2d_readvariableop_resource: 7
)conv2d_51_biasadd_readvariableop_resource: C
(conv2d_52_conv2d_readvariableop_resource: А8
)conv2d_52_biasadd_readvariableop_resource:	АD
(conv2d_53_conv2d_readvariableop_resource:АА8
)conv2d_53_biasadd_readvariableop_resource:	А<
'dense_42_matmul_readvariableop_resource:АвА7
(dense_42_biasadd_readvariableop_resource:	А;
'dense_43_matmul_readvariableop_resource:
АА7
(dense_43_biasadd_readvariableop_resource:	А
identityИв6batch_normalization_17/FusedBatchNormV3/ReadVariableOpв8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_17/ReadVariableOpв'batch_normalization_17/ReadVariableOp_1в conv2d_51/BiasAdd/ReadVariableOpвconv2d_51/Conv2D/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв conv2d_52/BiasAdd/ReadVariableOpвconv2d_52/Conv2D/ReadVariableOpв conv2d_53/BiasAdd/ReadVariableOpвconv2d_53/Conv2D/ReadVariableOpвdense_42/BiasAdd/ReadVariableOpвdense_42/MatMul/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpвdense_43/BiasAdd/ReadVariableOpвdense_43/MatMul/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpЧ
lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
lambda_17/strided_slice/stackЫ
lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2!
lambda_17/strided_slice/stack_1Ы
lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2!
lambda_17/strided_slice/stack_2╕
lambda_17/strided_sliceStridedSlicelambda_17_input&lambda_17/strided_slice/stack:output:0(lambda_17/strided_slice/stack_1:output:0(lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_17/strided_slice╣
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_17/ReadVariableOp┐
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_17/ReadVariableOp_1ь
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ю
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 lambda_17/strided_slice:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3│
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOpц
conv2d_51/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_51/Conv2Dк
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp░
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_51/Relu╩
max_pooling2d_51/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_51/MaxPool┤
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_52/Conv2D/ReadVariableOp▌
conv2d_52/Conv2DConv2D!max_pooling2d_51/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_52/Conv2Dл
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp▒
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_52/BiasAdd
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_52/Relu╦
max_pooling2d_52/MaxPoolMaxPoolconv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_52/MaxPool╡
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_53/Conv2D/ReadVariableOp▌
conv2d_53/Conv2DConv2D!max_pooling2d_52/MaxPool:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_53/Conv2Dл
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp▒
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_53/Relu╦
max_pooling2d_53/MaxPoolMaxPoolconv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_53/MaxPoolФ
dropout_51/IdentityIdentity!max_pooling2d_53/MaxPool:output:0*
T0*0
_output_shapes
:         		А2
dropout_51/Identityu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_17/Constа
flatten_17/ReshapeReshapedropout_51/Identity:output:0flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_17/Reshapeл
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02 
dense_42/MatMul/ReadVariableOpд
dense_42/MatMulMatMulflatten_17/Reshape:output:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_42/MatMulи
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_42/BiasAdd/ReadVariableOpж
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_42/BiasAddt
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_42/ReluЖ
dropout_52/IdentityIdentitydense_42/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_52/Identityк
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_43/MatMul/ReadVariableOpе
dense_43/MatMulMatMuldropout_52/Identity:output:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_43/MatMulи
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_43/BiasAdd/ReadVariableOpж
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_43/BiasAddt
dense_43/ReluReludense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_43/ReluЖ
dropout_53/IdentityIdentitydense_43/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_53/Identity┘
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul╤
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╨
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulй
IdentityIdentitydropout_53/Identity:output:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp:` \
/
_output_shapes
:         KK
)
_user_specified_namelambda_17_input
Н
Ю
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148790

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
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
н
i
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2143766

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
┼
Ю
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148826

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
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
р
N
2__inference_max_pooling2d_49_layer_call_fn_2143784

inputs
identityє
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_21437782
PartitionedCallП
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
─
b
F__inference_lambda_17_layer_call_and_return_conditional_losses_2145080

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
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
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ў
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_2148532

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╟
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
┴
┬
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2143700

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
р
N
2__inference_max_pooling2d_51_layer_call_fn_2144642

inputs
identityє
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_21446362
PartitionedCallП
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
┴
┬
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148397

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
╙
г
+__inference_conv2d_50_layer_call_fn_2148494

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_50_layer_call_and_return_conditional_losses_21438932
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╠
а
+__inference_conv2d_51_layer_call_fn_2148859

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallГ
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
GPU2 *0J 8В *O
fJRH
F__inference_conv2d_51_layer_call_and_return_conditional_losses_21447272
StatefulPartitionedCallЦ
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
╦
H
,__inference_dropout_49_layer_call_fn_2148580

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_21439432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ь
╡
__inference_loss_fn_1_2148683O
:dense_40_kernel_regularizer_square_readvariableop_resource:АвА
identityИв1dense_40/kernel/Regularizer/Square/ReadVariableOpф
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_40_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulЪ
IdentityIdentity#dense_40/kernel/Regularizer/mul:z:02^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp
∙
┬
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148433

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
└
┤
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2143857

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu╧
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_48/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╓
 
/__inference_sequential_16_layer_call_fn_2147332

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallЭ
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
:         А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_21443122
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
└
┤
F__inference_conv2d_51_layer_call_and_return_conditional_losses_2144727

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu╧
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
н
i
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_2144660

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
╢
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_2148661

inputs
identityИc
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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
└
┤
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2148465

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu╧
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_48/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ё
╙
8__inference_batch_normalization_17_layer_call_fn_2148746

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall║
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_21445702
StatefulPartitionedCallи
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
▌
H
,__inference_flatten_17_layer_call_fn_2148948

inputs
identity╠
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_17_layer_call_and_return_conditional_losses_21447832
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         Ав2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
к
╙
8__inference_batch_normalization_17_layer_call_fn_2148759

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallк
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_21447002
StatefulPartitionedCallЦ
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
Д
╙
%__inference_signature_wrapper_2146187
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: %

unknown_19: А

unknown_20:	А&

unknown_21:АА

unknown_22:	А

unknown_23:АвА

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCall╫
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
:         *@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_21436342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
н
i
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2144648

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
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
Є
╙
8__inference_batch_normalization_17_layer_call_fn_2148733

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall╝
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_21445262
StatefulPartitionedCallи
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
╢
f
G__inference_dropout_53_layer_call_and_return_conditional_losses_2149072

inputs
identityИc
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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ё
И
/__inference_sequential_17_layer_call_fn_2147889
lambda_17_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCalllambda_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_21451822
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         KK
)
_user_specified_namelambda_17_input
┴
┬
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148808

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
AssignNewValue_1Р
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
╦
H
,__inference_dropout_50_layer_call_fn_2148639

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_21439732
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╠
а
+__inference_conv2d_48_layer_call_fn_2148448

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallГ
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
GPU2 *0J 8В *O
fJRH
F__inference_conv2d_48_layer_call_and_return_conditional_losses_21438572
StatefulPartitionedCallЦ
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
└
о
E__inference_dense_42_layer_call_and_return_conditional_losses_2148986

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpР
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╚
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
щ
┤
__inference_loss_fn_5_2149105N
:dense_43_kernel_regularizer_square_readvariableop_resource:
АА
identityИв1dense_43/kernel/Regularizer/Square/ReadVariableOpу
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_43_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulЪ
IdentityIdentity#dense_43/kernel/Regularizer/mul:z:02^dense_43/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp
╓
 
/__inference_sequential_17_layer_call_fn_2147856

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallЭ
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
:         А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_21451822
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╢
f
G__inference_dropout_53_layer_call_and_return_conditional_losses_2144915

inputs
identityИc
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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
р
N
2__inference_max_pooling2d_53_layer_call_fn_2144666

inputs
identityє
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_21446602
PartitionedCallП
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
ЖЫ
·
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148076

inputs<
.batch_normalization_17_readvariableop_resource:>
0batch_normalization_17_readvariableop_1_resource:M
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_51_conv2d_readvariableop_resource: 7
)conv2d_51_biasadd_readvariableop_resource: C
(conv2d_52_conv2d_readvariableop_resource: А8
)conv2d_52_biasadd_readvariableop_resource:	АD
(conv2d_53_conv2d_readvariableop_resource:АА8
)conv2d_53_biasadd_readvariableop_resource:	А<
'dense_42_matmul_readvariableop_resource:АвА7
(dense_42_biasadd_readvariableop_resource:	А;
'dense_43_matmul_readvariableop_resource:
АА7
(dense_43_biasadd_readvariableop_resource:	А
identityИв%batch_normalization_17/AssignNewValueв'batch_normalization_17/AssignNewValue_1в6batch_normalization_17/FusedBatchNormV3/ReadVariableOpв8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_17/ReadVariableOpв'batch_normalization_17/ReadVariableOp_1в conv2d_51/BiasAdd/ReadVariableOpвconv2d_51/Conv2D/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв conv2d_52/BiasAdd/ReadVariableOpвconv2d_52/Conv2D/ReadVariableOpв conv2d_53/BiasAdd/ReadVariableOpвconv2d_53/Conv2D/ReadVariableOpвdense_42/BiasAdd/ReadVariableOpвdense_42/MatMul/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpвdense_43/BiasAdd/ReadVariableOpвdense_43/MatMul/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpЧ
lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
lambda_17/strided_slice/stackЫ
lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2!
lambda_17/strided_slice/stack_1Ы
lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2!
lambda_17/strided_slice/stack_2п
lambda_17/strided_sliceStridedSliceinputs&lambda_17/strided_slice/stack:output:0(lambda_17/strided_slice/stack_1:output:0(lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_17/strided_slice╣
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_17/ReadVariableOp┐
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_17/ReadVariableOp_1ь
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1№
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 lambda_17/strided_slice:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_17/FusedBatchNormV3╡
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue┴
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1│
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOpц
conv2d_51/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_51/Conv2Dк
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp░
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_51/Relu╩
max_pooling2d_51/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_51/MaxPool┤
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_52/Conv2D/ReadVariableOp▌
conv2d_52/Conv2DConv2D!max_pooling2d_51/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_52/Conv2Dл
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp▒
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_52/BiasAdd
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_52/Relu╦
max_pooling2d_52/MaxPoolMaxPoolconv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_52/MaxPool╡
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_53/Conv2D/ReadVariableOp▌
conv2d_53/Conv2DConv2D!max_pooling2d_52/MaxPool:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_53/Conv2Dл
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp▒
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_53/Relu╦
max_pooling2d_53/MaxPoolMaxPoolconv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_53/MaxPooly
dropout_51/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_51/dropout/Const╕
dropout_51/dropout/MulMul!max_pooling2d_53/MaxPool:output:0!dropout_51/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout_51/dropout/MulЕ
dropout_51/dropout/ShapeShape!max_pooling2d_53/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_51/dropout/Shape▐
/dropout_51/dropout/random_uniform/RandomUniformRandomUniform!dropout_51/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype021
/dropout_51/dropout/random_uniform/RandomUniformЛ
!dropout_51/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_51/dropout/GreaterEqual/yє
dropout_51/dropout/GreaterEqualGreaterEqual8dropout_51/dropout/random_uniform/RandomUniform:output:0*dropout_51/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2!
dropout_51/dropout/GreaterEqualй
dropout_51/dropout/CastCast#dropout_51/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout_51/dropout/Castп
dropout_51/dropout/Mul_1Muldropout_51/dropout/Mul:z:0dropout_51/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout_51/dropout/Mul_1u
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_17/Constа
flatten_17/ReshapeReshapedropout_51/dropout/Mul_1:z:0flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_17/Reshapeл
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02 
dense_42/MatMul/ReadVariableOpд
dense_42/MatMulMatMulflatten_17/Reshape:output:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_42/MatMulи
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_42/BiasAdd/ReadVariableOpж
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_42/BiasAddt
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_42/Reluy
dropout_52/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_52/dropout/Constк
dropout_52/dropout/MulMuldense_42/Relu:activations:0!dropout_52/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_52/dropout/Mul
dropout_52/dropout/ShapeShapedense_42/Relu:activations:0*
T0*
_output_shapes
:2
dropout_52/dropout/Shape╓
/dropout_52/dropout/random_uniform/RandomUniformRandomUniform!dropout_52/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_52/dropout/random_uniform/RandomUniformЛ
!dropout_52/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_52/dropout/GreaterEqual/yы
dropout_52/dropout/GreaterEqualGreaterEqual8dropout_52/dropout/random_uniform/RandomUniform:output:0*dropout_52/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_52/dropout/GreaterEqualб
dropout_52/dropout/CastCast#dropout_52/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_52/dropout/Castз
dropout_52/dropout/Mul_1Muldropout_52/dropout/Mul:z:0dropout_52/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_52/dropout/Mul_1к
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_43/MatMul/ReadVariableOpе
dense_43/MatMulMatMuldropout_52/dropout/Mul_1:z:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_43/MatMulи
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_43/BiasAdd/ReadVariableOpж
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_43/BiasAddt
dense_43/ReluReludense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_43/Reluy
dropout_53/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_53/dropout/Constк
dropout_53/dropout/MulMuldense_43/Relu:activations:0!dropout_53/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_53/dropout/Mul
dropout_53/dropout/ShapeShapedense_43/Relu:activations:0*
T0*
_output_shapes
:2
dropout_53/dropout/Shape╓
/dropout_53/dropout/random_uniform/RandomUniformRandomUniform!dropout_53/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_53/dropout/random_uniform/RandomUniformЛ
!dropout_53/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_53/dropout/GreaterEqual/yы
dropout_53/dropout/GreaterEqualGreaterEqual8dropout_53/dropout/random_uniform/RandomUniform:output:0*dropout_53/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_53/dropout/GreaterEqualб
dropout_53/dropout/CastCast#dropout_53/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_53/dropout/Castз
dropout_53/dropout/Mul_1Muldropout_53/dropout/Mul:z:0dropout_53/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_53/dropout/Mul_1┘
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul╤
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╨
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mul√
IdentityIdentitydropout_53/dropout/Mul_1:z:0&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
в
В
F__inference_conv2d_53_layer_call_and_return_conditional_losses_2144763

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╢
f
G__inference_dropout_52_layer_call_and_return_conditional_losses_2144948

inputs
identityИc
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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Н 
Е#
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2146831

inputsJ
<sequential_16_batch_normalization_16_readvariableop_resource:L
>sequential_16_batch_normalization_16_readvariableop_1_resource:[
Msequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:]
Osequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_16_conv2d_48_conv2d_readvariableop_resource: E
7sequential_16_conv2d_48_biasadd_readvariableop_resource: Q
6sequential_16_conv2d_49_conv2d_readvariableop_resource: АF
7sequential_16_conv2d_49_biasadd_readvariableop_resource:	АR
6sequential_16_conv2d_50_conv2d_readvariableop_resource:ААF
7sequential_16_conv2d_50_biasadd_readvariableop_resource:	АJ
5sequential_16_dense_40_matmul_readvariableop_resource:АвАE
6sequential_16_dense_40_biasadd_readvariableop_resource:	АI
5sequential_16_dense_41_matmul_readvariableop_resource:
ААE
6sequential_16_dense_41_biasadd_readvariableop_resource:	АJ
<sequential_17_batch_normalization_17_readvariableop_resource:L
>sequential_17_batch_normalization_17_readvariableop_1_resource:[
Msequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_17_conv2d_51_conv2d_readvariableop_resource: E
7sequential_17_conv2d_51_biasadd_readvariableop_resource: Q
6sequential_17_conv2d_52_conv2d_readvariableop_resource: АF
7sequential_17_conv2d_52_biasadd_readvariableop_resource:	АR
6sequential_17_conv2d_53_conv2d_readvariableop_resource:ААF
7sequential_17_conv2d_53_biasadd_readvariableop_resource:	АJ
5sequential_17_dense_42_matmul_readvariableop_resource:АвАE
6sequential_17_dense_42_biasadd_readvariableop_resource:	АI
5sequential_17_dense_43_matmul_readvariableop_resource:
ААE
6sequential_17_dense_43_biasadd_readvariableop_resource:	А:
'dense_44_matmul_readvariableop_resource:	А6
(dense_44_biasadd_readvariableop_resource:
identityИв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpв3sequential_16/batch_normalization_16/AssignNewValueв5sequential_16/batch_normalization_16/AssignNewValue_1вDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpвFsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в3sequential_16/batch_normalization_16/ReadVariableOpв5sequential_16/batch_normalization_16/ReadVariableOp_1в.sequential_16/conv2d_48/BiasAdd/ReadVariableOpв-sequential_16/conv2d_48/Conv2D/ReadVariableOpв.sequential_16/conv2d_49/BiasAdd/ReadVariableOpв-sequential_16/conv2d_49/Conv2D/ReadVariableOpв.sequential_16/conv2d_50/BiasAdd/ReadVariableOpв-sequential_16/conv2d_50/Conv2D/ReadVariableOpв-sequential_16/dense_40/BiasAdd/ReadVariableOpв,sequential_16/dense_40/MatMul/ReadVariableOpв-sequential_16/dense_41/BiasAdd/ReadVariableOpв,sequential_16/dense_41/MatMul/ReadVariableOpв3sequential_17/batch_normalization_17/AssignNewValueв5sequential_17/batch_normalization_17/AssignNewValue_1вDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpвFsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в3sequential_17/batch_normalization_17/ReadVariableOpв5sequential_17/batch_normalization_17/ReadVariableOp_1в.sequential_17/conv2d_51/BiasAdd/ReadVariableOpв-sequential_17/conv2d_51/Conv2D/ReadVariableOpв.sequential_17/conv2d_52/BiasAdd/ReadVariableOpв-sequential_17/conv2d_52/Conv2D/ReadVariableOpв.sequential_17/conv2d_53/BiasAdd/ReadVariableOpв-sequential_17/conv2d_53/Conv2D/ReadVariableOpв-sequential_17/dense_42/BiasAdd/ReadVariableOpв,sequential_17/dense_42/MatMul/ReadVariableOpв-sequential_17/dense_43/BiasAdd/ReadVariableOpв,sequential_17/dense_43/MatMul/ReadVariableOp│
+sequential_16/lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_16/lambda_16/strided_slice/stack╖
-sequential_16/lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2/
-sequential_16/lambda_16/strided_slice/stack_1╖
-sequential_16/lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_16/lambda_16/strided_slice/stack_2ї
%sequential_16/lambda_16/strided_sliceStridedSliceinputs4sequential_16/lambda_16/strided_slice/stack:output:06sequential_16/lambda_16/strided_slice/stack_1:output:06sequential_16/lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_16/lambda_16/strided_sliceу
3sequential_16/batch_normalization_16/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_16/batch_normalization_16/ReadVariableOpщ
5sequential_16/batch_normalization_16/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_16/batch_normalization_16/ReadVariableOp_1Ц
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1▐
5sequential_16/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3.sequential_16/lambda_16/strided_slice:output:0;sequential_16/batch_normalization_16/ReadVariableOp:value:0=sequential_16/batch_normalization_16/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<27
5sequential_16/batch_normalization_16/FusedBatchNormV3√
3sequential_16/batch_normalization_16/AssignNewValueAssignVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resourceBsequential_16/batch_normalization_16/FusedBatchNormV3:batch_mean:0E^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3sequential_16/batch_normalization_16/AssignNewValueЗ
5sequential_16/batch_normalization_16/AssignNewValue_1AssignVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resourceFsequential_16/batch_normalization_16/FusedBatchNormV3:batch_variance:0G^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5sequential_16/batch_normalization_16/AssignNewValue_1▌
-sequential_16/conv2d_48/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_48/Conv2D/ReadVariableOpЮ
sequential_16/conv2d_48/Conv2DConv2D9sequential_16/batch_normalization_16/FusedBatchNormV3:y:05sequential_16/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_16/conv2d_48/Conv2D╘
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpш
sequential_16/conv2d_48/BiasAddBiasAdd'sequential_16/conv2d_48/Conv2D:output:06sequential_16/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_16/conv2d_48/BiasAddи
sequential_16/conv2d_48/ReluRelu(sequential_16/conv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_16/conv2d_48/ReluЇ
&sequential_16/max_pooling2d_48/MaxPoolMaxPool*sequential_16/conv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool▐
-sequential_16/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_16/conv2d_49/Conv2D/ReadVariableOpХ
sequential_16/conv2d_49/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_16/conv2d_49/Conv2D╒
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_49/BiasAddBiasAdd'sequential_16/conv2d_49/Conv2D:output:06sequential_16/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_16/conv2d_49/BiasAddй
sequential_16/conv2d_49/ReluRelu(sequential_16/conv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_16/conv2d_49/Reluї
&sequential_16/max_pooling2d_49/MaxPoolMaxPool*sequential_16/conv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool▀
-sequential_16/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_16/conv2d_50/Conv2D/ReadVariableOpХ
sequential_16/conv2d_50/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:05sequential_16/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_16/conv2d_50/Conv2D╒
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_50/BiasAddBiasAdd'sequential_16/conv2d_50/Conv2D:output:06sequential_16/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_16/conv2d_50/BiasAddй
sequential_16/conv2d_50/ReluRelu(sequential_16/conv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_16/conv2d_50/Reluї
&sequential_16/max_pooling2d_50/MaxPoolMaxPool*sequential_16/conv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPoolХ
&sequential_16/dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2(
&sequential_16/dropout_48/dropout/ConstЁ
$sequential_16/dropout_48/dropout/MulMul/sequential_16/max_pooling2d_50/MaxPool:output:0/sequential_16/dropout_48/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2&
$sequential_16/dropout_48/dropout/Mulп
&sequential_16/dropout_48/dropout/ShapeShape/sequential_16/max_pooling2d_50/MaxPool:output:0*
T0*
_output_shapes
:2(
&sequential_16/dropout_48/dropout/ShapeИ
=sequential_16/dropout_48/dropout/random_uniform/RandomUniformRandomUniform/sequential_16/dropout_48/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02?
=sequential_16/dropout_48/dropout/random_uniform/RandomUniformз
/sequential_16/dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=21
/sequential_16/dropout_48/dropout/GreaterEqual/yл
-sequential_16/dropout_48/dropout/GreaterEqualGreaterEqualFsequential_16/dropout_48/dropout/random_uniform/RandomUniform:output:08sequential_16/dropout_48/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2/
-sequential_16/dropout_48/dropout/GreaterEqual╙
%sequential_16/dropout_48/dropout/CastCast1sequential_16/dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2'
%sequential_16/dropout_48/dropout/Castч
&sequential_16/dropout_48/dropout/Mul_1Mul(sequential_16/dropout_48/dropout/Mul:z:0)sequential_16/dropout_48/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2(
&sequential_16/dropout_48/dropout/Mul_1С
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_16/flatten_16/Const╪
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_48/dropout/Mul_1:z:0'sequential_16/flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_16/flatten_16/Reshape╒
,sequential_16/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_16/dense_40/MatMul/ReadVariableOp▄
sequential_16/dense_40/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/MatMul╥
-sequential_16/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_40/BiasAdd/ReadVariableOp▐
sequential_16/dense_40/BiasAddBiasAdd'sequential_16/dense_40/MatMul:product:05sequential_16/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_40/BiasAddЮ
sequential_16/dense_40/ReluRelu'sequential_16/dense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/ReluХ
&sequential_16/dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&sequential_16/dropout_49/dropout/Constт
$sequential_16/dropout_49/dropout/MulMul)sequential_16/dense_40/Relu:activations:0/sequential_16/dropout_49/dropout/Const:output:0*
T0*(
_output_shapes
:         А2&
$sequential_16/dropout_49/dropout/Mulй
&sequential_16/dropout_49/dropout/ShapeShape)sequential_16/dense_40/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_16/dropout_49/dropout/ShapeА
=sequential_16/dropout_49/dropout/random_uniform/RandomUniformRandomUniform/sequential_16/dropout_49/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02?
=sequential_16/dropout_49/dropout/random_uniform/RandomUniformз
/sequential_16/dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/sequential_16/dropout_49/dropout/GreaterEqual/yг
-sequential_16/dropout_49/dropout/GreaterEqualGreaterEqualFsequential_16/dropout_49/dropout/random_uniform/RandomUniform:output:08sequential_16/dropout_49/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2/
-sequential_16/dropout_49/dropout/GreaterEqual╦
%sequential_16/dropout_49/dropout/CastCast1sequential_16/dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2'
%sequential_16/dropout_49/dropout/Cast▀
&sequential_16/dropout_49/dropout/Mul_1Mul(sequential_16/dropout_49/dropout/Mul:z:0)sequential_16/dropout_49/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2(
&sequential_16/dropout_49/dropout/Mul_1╘
,sequential_16/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_16/dense_41/MatMul/ReadVariableOp▌
sequential_16/dense_41/MatMulMatMul*sequential_16/dropout_49/dropout/Mul_1:z:04sequential_16/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/MatMul╥
-sequential_16/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_41/BiasAdd/ReadVariableOp▐
sequential_16/dense_41/BiasAddBiasAdd'sequential_16/dense_41/MatMul:product:05sequential_16/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_41/BiasAddЮ
sequential_16/dense_41/ReluRelu'sequential_16/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/ReluХ
&sequential_16/dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&sequential_16/dropout_50/dropout/Constт
$sequential_16/dropout_50/dropout/MulMul)sequential_16/dense_41/Relu:activations:0/sequential_16/dropout_50/dropout/Const:output:0*
T0*(
_output_shapes
:         А2&
$sequential_16/dropout_50/dropout/Mulй
&sequential_16/dropout_50/dropout/ShapeShape)sequential_16/dense_41/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_16/dropout_50/dropout/ShapeА
=sequential_16/dropout_50/dropout/random_uniform/RandomUniformRandomUniform/sequential_16/dropout_50/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02?
=sequential_16/dropout_50/dropout/random_uniform/RandomUniformз
/sequential_16/dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/sequential_16/dropout_50/dropout/GreaterEqual/yг
-sequential_16/dropout_50/dropout/GreaterEqualGreaterEqualFsequential_16/dropout_50/dropout/random_uniform/RandomUniform:output:08sequential_16/dropout_50/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2/
-sequential_16/dropout_50/dropout/GreaterEqual╦
%sequential_16/dropout_50/dropout/CastCast1sequential_16/dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2'
%sequential_16/dropout_50/dropout/Cast▀
&sequential_16/dropout_50/dropout/Mul_1Mul(sequential_16/dropout_50/dropout/Mul:z:0)sequential_16/dropout_50/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2(
&sequential_16/dropout_50/dropout/Mul_1│
+sequential_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2-
+sequential_17/lambda_17/strided_slice/stack╖
-sequential_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2/
-sequential_17/lambda_17/strided_slice/stack_1╖
-sequential_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_17/lambda_17/strided_slice/stack_2ї
%sequential_17/lambda_17/strided_sliceStridedSliceinputs4sequential_17/lambda_17/strided_slice/stack:output:06sequential_17/lambda_17/strided_slice/stack_1:output:06sequential_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_17/lambda_17/strided_sliceу
3sequential_17/batch_normalization_17/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_17/batch_normalization_17/ReadVariableOpщ
5sequential_17/batch_normalization_17/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_17/batch_normalization_17/ReadVariableOp_1Ц
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1▐
5sequential_17/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3.sequential_17/lambda_17/strided_slice:output:0;sequential_17/batch_normalization_17/ReadVariableOp:value:0=sequential_17/batch_normalization_17/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<27
5sequential_17/batch_normalization_17/FusedBatchNormV3√
3sequential_17/batch_normalization_17/AssignNewValueAssignVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resourceBsequential_17/batch_normalization_17/FusedBatchNormV3:batch_mean:0E^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3sequential_17/batch_normalization_17/AssignNewValueЗ
5sequential_17/batch_normalization_17/AssignNewValue_1AssignVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resourceFsequential_17/batch_normalization_17/FusedBatchNormV3:batch_variance:0G^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5sequential_17/batch_normalization_17/AssignNewValue_1▌
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOpЮ
sequential_17/conv2d_51/Conv2DConv2D9sequential_17/batch_normalization_17/FusedBatchNormV3:y:05sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_17/conv2d_51/Conv2D╘
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpш
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_17/conv2d_51/BiasAddи
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_17/conv2d_51/ReluЇ
&sequential_17/max_pooling2d_51/MaxPoolMaxPool*sequential_17/conv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_51/MaxPool▐
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOpХ
sequential_17/conv2d_52/Conv2DConv2D/sequential_17/max_pooling2d_51/MaxPool:output:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_17/conv2d_52/Conv2D╒
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_17/conv2d_52/BiasAddй
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_17/conv2d_52/Reluї
&sequential_17/max_pooling2d_52/MaxPoolMaxPool*sequential_17/conv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_52/MaxPool▀
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOpХ
sequential_17/conv2d_53/Conv2DConv2D/sequential_17/max_pooling2d_52/MaxPool:output:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_17/conv2d_53/Conv2D╒
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_17/conv2d_53/BiasAddй
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_17/conv2d_53/Reluї
&sequential_17/max_pooling2d_53/MaxPoolMaxPool*sequential_17/conv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_53/MaxPoolХ
&sequential_17/dropout_51/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2(
&sequential_17/dropout_51/dropout/ConstЁ
$sequential_17/dropout_51/dropout/MulMul/sequential_17/max_pooling2d_53/MaxPool:output:0/sequential_17/dropout_51/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2&
$sequential_17/dropout_51/dropout/Mulп
&sequential_17/dropout_51/dropout/ShapeShape/sequential_17/max_pooling2d_53/MaxPool:output:0*
T0*
_output_shapes
:2(
&sequential_17/dropout_51/dropout/ShapeИ
=sequential_17/dropout_51/dropout/random_uniform/RandomUniformRandomUniform/sequential_17/dropout_51/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02?
=sequential_17/dropout_51/dropout/random_uniform/RandomUniformз
/sequential_17/dropout_51/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=21
/sequential_17/dropout_51/dropout/GreaterEqual/yл
-sequential_17/dropout_51/dropout/GreaterEqualGreaterEqualFsequential_17/dropout_51/dropout/random_uniform/RandomUniform:output:08sequential_17/dropout_51/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2/
-sequential_17/dropout_51/dropout/GreaterEqual╙
%sequential_17/dropout_51/dropout/CastCast1sequential_17/dropout_51/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2'
%sequential_17/dropout_51/dropout/Castч
&sequential_17/dropout_51/dropout/Mul_1Mul(sequential_17/dropout_51/dropout/Mul:z:0)sequential_17/dropout_51/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2(
&sequential_17/dropout_51/dropout/Mul_1С
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_17/flatten_17/Const╪
 sequential_17/flatten_17/ReshapeReshape*sequential_17/dropout_51/dropout/Mul_1:z:0'sequential_17/flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_17/flatten_17/Reshape╒
,sequential_17/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_17/dense_42/MatMul/ReadVariableOp▄
sequential_17/dense_42/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/MatMul╥
-sequential_17/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_42/BiasAdd/ReadVariableOp▐
sequential_17/dense_42/BiasAddBiasAdd'sequential_17/dense_42/MatMul:product:05sequential_17/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_42/BiasAddЮ
sequential_17/dense_42/ReluRelu'sequential_17/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/ReluХ
&sequential_17/dropout_52/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&sequential_17/dropout_52/dropout/Constт
$sequential_17/dropout_52/dropout/MulMul)sequential_17/dense_42/Relu:activations:0/sequential_17/dropout_52/dropout/Const:output:0*
T0*(
_output_shapes
:         А2&
$sequential_17/dropout_52/dropout/Mulй
&sequential_17/dropout_52/dropout/ShapeShape)sequential_17/dense_42/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_17/dropout_52/dropout/ShapeА
=sequential_17/dropout_52/dropout/random_uniform/RandomUniformRandomUniform/sequential_17/dropout_52/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02?
=sequential_17/dropout_52/dropout/random_uniform/RandomUniformз
/sequential_17/dropout_52/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/sequential_17/dropout_52/dropout/GreaterEqual/yг
-sequential_17/dropout_52/dropout/GreaterEqualGreaterEqualFsequential_17/dropout_52/dropout/random_uniform/RandomUniform:output:08sequential_17/dropout_52/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2/
-sequential_17/dropout_52/dropout/GreaterEqual╦
%sequential_17/dropout_52/dropout/CastCast1sequential_17/dropout_52/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2'
%sequential_17/dropout_52/dropout/Cast▀
&sequential_17/dropout_52/dropout/Mul_1Mul(sequential_17/dropout_52/dropout/Mul:z:0)sequential_17/dropout_52/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2(
&sequential_17/dropout_52/dropout/Mul_1╘
,sequential_17/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_17/dense_43/MatMul/ReadVariableOp▌
sequential_17/dense_43/MatMulMatMul*sequential_17/dropout_52/dropout/Mul_1:z:04sequential_17/dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/MatMul╥
-sequential_17/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_43/BiasAdd/ReadVariableOp▐
sequential_17/dense_43/BiasAddBiasAdd'sequential_17/dense_43/MatMul:product:05sequential_17/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_43/BiasAddЮ
sequential_17/dense_43/ReluRelu'sequential_17/dense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/ReluХ
&sequential_17/dropout_53/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&sequential_17/dropout_53/dropout/Constт
$sequential_17/dropout_53/dropout/MulMul)sequential_17/dense_43/Relu:activations:0/sequential_17/dropout_53/dropout/Const:output:0*
T0*(
_output_shapes
:         А2&
$sequential_17/dropout_53/dropout/Mulй
&sequential_17/dropout_53/dropout/ShapeShape)sequential_17/dense_43/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_17/dropout_53/dropout/ShapeА
=sequential_17/dropout_53/dropout/random_uniform/RandomUniformRandomUniform/sequential_17/dropout_53/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02?
=sequential_17/dropout_53/dropout/random_uniform/RandomUniformз
/sequential_17/dropout_53/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/sequential_17/dropout_53/dropout/GreaterEqual/yг
-sequential_17/dropout_53/dropout/GreaterEqualGreaterEqualFsequential_17/dropout_53/dropout/random_uniform/RandomUniform:output:08sequential_17/dropout_53/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2/
-sequential_17/dropout_53/dropout/GreaterEqual╦
%sequential_17/dropout_53/dropout/CastCast1sequential_17/dropout_53/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2'
%sequential_17/dropout_53/dropout/Cast▀
&sequential_17/dropout_53/dropout/Mul_1Mul(sequential_17/dropout_53/dropout/Mul:z:0)sequential_17/dropout_53/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2(
&sequential_17/dropout_53/dropout/Mul_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╞
concatConcatV2*sequential_16/dropout_50/dropout/Mul_1:z:0*sequential_17/dropout_53/dropout/Mul_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatй
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_44/MatMul/ReadVariableOpЧ
dense_44/MatMulMatMulconcat:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpе
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/BiasAdd|
dense_44/SoftmaxSoftmaxdense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_44/Softmaxч
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul▀
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul▐
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulч
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul▀
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul▐
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulЕ
IdentityIdentitydense_44/Softmax:softmax:03^conv2d_48/kernel/Regularizer/Square/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp4^sequential_16/batch_normalization_16/AssignNewValue6^sequential_16/batch_normalization_16/AssignNewValue_1E^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_16/ReadVariableOp6^sequential_16/batch_normalization_16/ReadVariableOp_1/^sequential_16/conv2d_48/BiasAdd/ReadVariableOp.^sequential_16/conv2d_48/Conv2D/ReadVariableOp/^sequential_16/conv2d_49/BiasAdd/ReadVariableOp.^sequential_16/conv2d_49/Conv2D/ReadVariableOp/^sequential_16/conv2d_50/BiasAdd/ReadVariableOp.^sequential_16/conv2d_50/Conv2D/ReadVariableOp.^sequential_16/dense_40/BiasAdd/ReadVariableOp-^sequential_16/dense_40/MatMul/ReadVariableOp.^sequential_16/dense_41/BiasAdd/ReadVariableOp-^sequential_16/dense_41/MatMul/ReadVariableOp4^sequential_17/batch_normalization_17/AssignNewValue6^sequential_17/batch_normalization_17/AssignNewValue_1E^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_17/ReadVariableOp6^sequential_17/batch_normalization_17/ReadVariableOp_1/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_42/BiasAdd/ReadVariableOp-^sequential_17/dense_42/MatMul/ReadVariableOp.^sequential_17/dense_43/BiasAdd/ReadVariableOp-^sequential_17/dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2j
3sequential_16/batch_normalization_16/AssignNewValue3sequential_16/batch_normalization_16/AssignNewValue2n
5sequential_16/batch_normalization_16/AssignNewValue_15sequential_16/batch_normalization_16/AssignNewValue_12М
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_16/ReadVariableOp3sequential_16/batch_normalization_16/ReadVariableOp2n
5sequential_16/batch_normalization_16/ReadVariableOp_15sequential_16/batch_normalization_16/ReadVariableOp_12`
.sequential_16/conv2d_48/BiasAdd/ReadVariableOp.sequential_16/conv2d_48/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_48/Conv2D/ReadVariableOp-sequential_16/conv2d_48/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_49/BiasAdd/ReadVariableOp.sequential_16/conv2d_49/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_49/Conv2D/ReadVariableOp-sequential_16/conv2d_49/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_50/BiasAdd/ReadVariableOp.sequential_16/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_50/Conv2D/ReadVariableOp-sequential_16/conv2d_50/Conv2D/ReadVariableOp2^
-sequential_16/dense_40/BiasAdd/ReadVariableOp-sequential_16/dense_40/BiasAdd/ReadVariableOp2\
,sequential_16/dense_40/MatMul/ReadVariableOp,sequential_16/dense_40/MatMul/ReadVariableOp2^
-sequential_16/dense_41/BiasAdd/ReadVariableOp-sequential_16/dense_41/BiasAdd/ReadVariableOp2\
,sequential_16/dense_41/MatMul/ReadVariableOp,sequential_16/dense_41/MatMul/ReadVariableOp2j
3sequential_17/batch_normalization_17/AssignNewValue3sequential_17/batch_normalization_17/AssignNewValue2n
5sequential_17/batch_normalization_17/AssignNewValue_15sequential_17/batch_normalization_17/AssignNewValue_12М
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_17/ReadVariableOp3sequential_17/batch_normalization_17/ReadVariableOp2n
5sequential_17/batch_normalization_17/ReadVariableOp_15sequential_17/batch_normalization_17/ReadVariableOp_12`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_42/BiasAdd/ReadVariableOp-sequential_17/dense_42/BiasAdd/ReadVariableOp2\
,sequential_17/dense_42/MatMul/ReadVariableOp,sequential_17/dense_42/MatMul/ReadVariableOp2^
-sequential_17/dense_43/BiasAdd/ReadVariableOp-sequential_17/dense_43/BiasAdd/ReadVariableOp2\
,sequential_17/dense_43/MatMul/ReadVariableOp,sequential_17/dense_43/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
бЫ
Г
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148263
lambda_17_input<
.batch_normalization_17_readvariableop_resource:>
0batch_normalization_17_readvariableop_1_resource:M
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_51_conv2d_readvariableop_resource: 7
)conv2d_51_biasadd_readvariableop_resource: C
(conv2d_52_conv2d_readvariableop_resource: А8
)conv2d_52_biasadd_readvariableop_resource:	АD
(conv2d_53_conv2d_readvariableop_resource:АА8
)conv2d_53_biasadd_readvariableop_resource:	А<
'dense_42_matmul_readvariableop_resource:АвА7
(dense_42_biasadd_readvariableop_resource:	А;
'dense_43_matmul_readvariableop_resource:
АА7
(dense_43_biasadd_readvariableop_resource:	А
identityИв%batch_normalization_17/AssignNewValueв'batch_normalization_17/AssignNewValue_1в6batch_normalization_17/FusedBatchNormV3/ReadVariableOpв8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_17/ReadVariableOpв'batch_normalization_17/ReadVariableOp_1в conv2d_51/BiasAdd/ReadVariableOpвconv2d_51/Conv2D/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв conv2d_52/BiasAdd/ReadVariableOpвconv2d_52/Conv2D/ReadVariableOpв conv2d_53/BiasAdd/ReadVariableOpвconv2d_53/Conv2D/ReadVariableOpвdense_42/BiasAdd/ReadVariableOpвdense_42/MatMul/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpвdense_43/BiasAdd/ReadVariableOpвdense_43/MatMul/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpЧ
lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
lambda_17/strided_slice/stackЫ
lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2!
lambda_17/strided_slice/stack_1Ы
lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2!
lambda_17/strided_slice/stack_2╕
lambda_17/strided_sliceStridedSlicelambda_17_input&lambda_17/strided_slice/stack:output:0(lambda_17/strided_slice/stack_1:output:0(lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_17/strided_slice╣
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_17/ReadVariableOp┐
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_17/ReadVariableOp_1ь
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1№
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 lambda_17/strided_slice:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_17/FusedBatchNormV3╡
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue┴
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1│
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOpц
conv2d_51/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_51/Conv2Dк
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp░
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_51/Relu╩
max_pooling2d_51/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_51/MaxPool┤
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_52/Conv2D/ReadVariableOp▌
conv2d_52/Conv2DConv2D!max_pooling2d_51/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_52/Conv2Dл
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp▒
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_52/BiasAdd
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_52/Relu╦
max_pooling2d_52/MaxPoolMaxPoolconv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_52/MaxPool╡
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_53/Conv2D/ReadVariableOp▌
conv2d_53/Conv2DConv2D!max_pooling2d_52/MaxPool:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_53/Conv2Dл
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp▒
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_53/Relu╦
max_pooling2d_53/MaxPoolMaxPoolconv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_53/MaxPooly
dropout_51/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_51/dropout/Const╕
dropout_51/dropout/MulMul!max_pooling2d_53/MaxPool:output:0!dropout_51/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout_51/dropout/MulЕ
dropout_51/dropout/ShapeShape!max_pooling2d_53/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_51/dropout/Shape▐
/dropout_51/dropout/random_uniform/RandomUniformRandomUniform!dropout_51/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype021
/dropout_51/dropout/random_uniform/RandomUniformЛ
!dropout_51/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_51/dropout/GreaterEqual/yє
dropout_51/dropout/GreaterEqualGreaterEqual8dropout_51/dropout/random_uniform/RandomUniform:output:0*dropout_51/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2!
dropout_51/dropout/GreaterEqualй
dropout_51/dropout/CastCast#dropout_51/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout_51/dropout/Castп
dropout_51/dropout/Mul_1Muldropout_51/dropout/Mul:z:0dropout_51/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout_51/dropout/Mul_1u
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_17/Constа
flatten_17/ReshapeReshapedropout_51/dropout/Mul_1:z:0flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_17/Reshapeл
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02 
dense_42/MatMul/ReadVariableOpд
dense_42/MatMulMatMulflatten_17/Reshape:output:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_42/MatMulи
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_42/BiasAdd/ReadVariableOpж
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_42/BiasAddt
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_42/Reluy
dropout_52/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_52/dropout/Constк
dropout_52/dropout/MulMuldense_42/Relu:activations:0!dropout_52/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_52/dropout/Mul
dropout_52/dropout/ShapeShapedense_42/Relu:activations:0*
T0*
_output_shapes
:2
dropout_52/dropout/Shape╓
/dropout_52/dropout/random_uniform/RandomUniformRandomUniform!dropout_52/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_52/dropout/random_uniform/RandomUniformЛ
!dropout_52/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_52/dropout/GreaterEqual/yы
dropout_52/dropout/GreaterEqualGreaterEqual8dropout_52/dropout/random_uniform/RandomUniform:output:0*dropout_52/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_52/dropout/GreaterEqualб
dropout_52/dropout/CastCast#dropout_52/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_52/dropout/Castз
dropout_52/dropout/Mul_1Muldropout_52/dropout/Mul:z:0dropout_52/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_52/dropout/Mul_1к
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_43/MatMul/ReadVariableOpе
dense_43/MatMulMatMuldropout_52/dropout/Mul_1:z:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_43/MatMulи
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_43/BiasAdd/ReadVariableOpж
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_43/BiasAddt
dense_43/ReluReludense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_43/Reluy
dropout_53/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_53/dropout/Constк
dropout_53/dropout/MulMuldense_43/Relu:activations:0!dropout_53/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_53/dropout/Mul
dropout_53/dropout/ShapeShapedense_43/Relu:activations:0*
T0*
_output_shapes
:2
dropout_53/dropout/Shape╓
/dropout_53/dropout/random_uniform/RandomUniformRandomUniform!dropout_53/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_53/dropout/random_uniform/RandomUniformЛ
!dropout_53/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_53/dropout/GreaterEqual/yы
dropout_53/dropout/GreaterEqualGreaterEqual8dropout_53/dropout/random_uniform/RandomUniform:output:0*dropout_53/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_53/dropout/GreaterEqualб
dropout_53/dropout/CastCast#dropout_53/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_53/dropout/Castз
dropout_53/dropout/Mul_1Muldropout_53/dropout/Mul:z:0dropout_53/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_53/dropout/Mul_1┘
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul╤
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╨
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mul√
IdentityIdentitydropout_53/dropout/Mul_1:z:0&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp:` \
/
_output_shapes
:         KK
)
_user_specified_namelambda_17_input
°
e
G__inference_dropout_49_layer_call_and_return_conditional_losses_2143943

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
°
e
G__inference_dropout_50_layer_call_and_return_conditional_losses_2143973

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ў
f
G__inference_dropout_51_layer_call_and_return_conditional_losses_2148943

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╟
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
└
о
E__inference_dense_42_layer_call_and_return_conditional_losses_2144802

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpР
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╚
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
░a
О	
J__inference_sequential_16_layer_call_and_return_conditional_losses_2144312

inputs,
batch_normalization_16_2144252:,
batch_normalization_16_2144254:,
batch_normalization_16_2144256:,
batch_normalization_16_2144258:+
conv2d_48_2144261: 
conv2d_48_2144263: ,
conv2d_49_2144267: А 
conv2d_49_2144269:	А-
conv2d_50_2144273:АА 
conv2d_50_2144275:	А%
dense_40_2144281:АвА
dense_40_2144283:	А$
dense_41_2144287:
АА
dense_41_2144289:	А
identityИв.batch_normalization_16/StatefulPartitionedCallв!conv2d_48/StatefulPartitionedCallв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв!conv2d_49/StatefulPartitionedCallв!conv2d_50/StatefulPartitionedCallв dense_40/StatefulPartitionedCallв1dense_40/kernel/Regularizer/Square/ReadVariableOpв dense_41/StatefulPartitionedCallв1dense_41/kernel/Regularizer/Square/ReadVariableOpв"dropout_48/StatefulPartitionedCallв"dropout_49/StatefulPartitionedCallв"dropout_50/StatefulPartitionedCallх
lambda_16/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8В *O
fJRH
F__inference_lambda_16_layer_call_and_return_conditional_losses_21442102
lambda_16/PartitionedCall╚
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall"lambda_16/PartitionedCall:output:0batch_normalization_16_2144252batch_normalization_16_2144254batch_normalization_16_2144256batch_normalization_16_2144258*
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_214418320
.batch_normalization_16/StatefulPartitionedCall┌
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_48_2144261conv2d_48_2144263*
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
GPU2 *0J 8В *O
fJRH
F__inference_conv2d_48_layer_call_and_return_conditional_losses_21438572#
!conv2d_48/StatefulPartitionedCallЮ
 max_pooling2d_48/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_21437662"
 max_pooling2d_48/PartitionedCall═
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_49_2144267conv2d_49_2144269*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_49_layer_call_and_return_conditional_losses_21438752#
!conv2d_49/StatefulPartitionedCallЯ
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_21437782"
 max_pooling2d_49/PartitionedCall═
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_50_2144273conv2d_50_2144275*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_50_layer_call_and_return_conditional_losses_21438932#
!conv2d_50/StatefulPartitionedCallЯ
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_21437902"
 max_pooling2d_50/PartitionedCallд
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_21441172$
"dropout_48/StatefulPartitionedCallЗ
flatten_16/PartitionedCallPartitionedCall+dropout_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_16_layer_call_and_return_conditional_losses_21439132
flatten_16/PartitionedCall║
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_40_2144281dense_40_2144283*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_21439322"
 dense_40/StatefulPartitionedCall┴
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_21440782$
"dropout_49/StatefulPartitionedCall┬
 dense_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0dense_41_2144287dense_41_2144289*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_21439622"
 dense_41/StatefulPartitionedCall┴
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_21440452$
"dropout_50/StatefulPartitionedCall┬
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_48_2144261*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul║
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_2144281*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╣
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_41_2144287* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulя
IdentityIdentity+dropout_50/StatefulPartitionedCall:output:0/^batch_normalization_16/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall3^conv2d_48/kernel/Regularizer/Square/ReadVariableOp"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp!^dense_41/StatefulPartitionedCall2^dense_41/kernel/Regularizer/Square/ReadVariableOp#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▌
H
,__inference_flatten_16_layer_call_fn_2148537

inputs
identity╠
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_16_layer_call_and_return_conditional_losses_21439132
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         Ав2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
и
╪
*__inference_CNN_2jet_layer_call_fn_2146447
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: %

unknown_19: А

unknown_20:	А&

unknown_21:АА

unknown_22:	А

unknown_23:АвА

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCallЎ
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
:         *<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_21457402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
╠\
Я
J__inference_sequential_17_layer_call_and_return_conditional_losses_2144864

inputs,
batch_normalization_17_2144701:,
batch_normalization_17_2144703:,
batch_normalization_17_2144705:,
batch_normalization_17_2144707:+
conv2d_51_2144728: 
conv2d_51_2144730: ,
conv2d_52_2144746: А 
conv2d_52_2144748:	А-
conv2d_53_2144764:АА 
conv2d_53_2144766:	А%
dense_42_2144803:АвА
dense_42_2144805:	А$
dense_43_2144833:
АА
dense_43_2144835:	А
identityИв.batch_normalization_17/StatefulPartitionedCallв!conv2d_51/StatefulPartitionedCallв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв!conv2d_52/StatefulPartitionedCallв!conv2d_53/StatefulPartitionedCallв dense_42/StatefulPartitionedCallв1dense_42/kernel/Regularizer/Square/ReadVariableOpв dense_43/StatefulPartitionedCallв1dense_43/kernel/Regularizer/Square/ReadVariableOpх
lambda_17/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8В *O
fJRH
F__inference_lambda_17_layer_call_and_return_conditional_losses_21446812
lambda_17/PartitionedCall╩
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall"lambda_17/PartitionedCall:output:0batch_normalization_17_2144701batch_normalization_17_2144703batch_normalization_17_2144705batch_normalization_17_2144707*
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_214470020
.batch_normalization_17/StatefulPartitionedCall┌
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0conv2d_51_2144728conv2d_51_2144730*
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
GPU2 *0J 8В *O
fJRH
F__inference_conv2d_51_layer_call_and_return_conditional_losses_21447272#
!conv2d_51/StatefulPartitionedCallЮ
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_21446362"
 max_pooling2d_51/PartitionedCall═
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0conv2d_52_2144746conv2d_52_2144748*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_52_layer_call_and_return_conditional_losses_21447452#
!conv2d_52/StatefulPartitionedCallЯ
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_21446482"
 max_pooling2d_52/PartitionedCall═
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0conv2d_53_2144764conv2d_53_2144766*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_53_layer_call_and_return_conditional_losses_21447632#
!conv2d_53/StatefulPartitionedCallЯ
 max_pooling2d_53/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_21446602"
 max_pooling2d_53/PartitionedCallМ
dropout_51/PartitionedCallPartitionedCall)max_pooling2d_53/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_51_layer_call_and_return_conditional_losses_21447752
dropout_51/PartitionedCall 
flatten_17/PartitionedCallPartitionedCall#dropout_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_17_layer_call_and_return_conditional_losses_21447832
flatten_17/PartitionedCall║
 dense_42/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_42_2144803dense_42_2144805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_21448022"
 dense_42/StatefulPartitionedCallД
dropout_52/PartitionedCallPartitionedCall)dense_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_52_layer_call_and_return_conditional_losses_21448132
dropout_52/PartitionedCall║
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_52/PartitionedCall:output:0dense_43_2144833dense_43_2144835*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_21448322"
 dense_43/StatefulPartitionedCallД
dropout_53/PartitionedCallPartitionedCall)dense_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_21448432
dropout_53/PartitionedCall┬
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_51_2144728*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul║
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_42_2144803*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul╣
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_43_2144833* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mul°
IdentityIdentity#dropout_53/PartitionedCall:output:0/^batch_normalization_17/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall2^dense_42/kernel/Regularizer/Square/ReadVariableOp!^dense_43/StatefulPartitionedCall2^dense_43/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ў
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_2144117

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╟
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
в
В
F__inference_conv2d_53_layer_call_and_return_conditional_losses_2148916

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
м
Ы
*__inference_dense_42_layer_call_fn_2148969

inputs
unknown:АвА
	unknown_0:	А
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_21448022
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
∙
┬
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2145053

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
°
e
G__inference_dropout_53_layer_call_and_return_conditional_losses_2149060

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ў
f
G__inference_dropout_51_layer_call_and_return_conditional_losses_2144987

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╟
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
э
c
G__inference_flatten_16_layer_call_and_return_conditional_losses_2148543

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         Ав2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         Ав2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
э
c
G__inference_flatten_17_layer_call_and_return_conditional_losses_2148954

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         Ав2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         Ав2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╢
f
G__inference_dropout_52_layer_call_and_return_conditional_losses_2149013

inputs
identityИc
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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
р
N
2__inference_max_pooling2d_50_layer_call_fn_2143796

inputs
identityє
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_21437902
PartitionedCallП
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
э
c
G__inference_flatten_17_layer_call_and_return_conditional_losses_2144783

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         Ав2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         Ав2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
Ы
╝
__inference_loss_fn_0_2148672U
;conv2d_48_kernel_regularizer_square_readvariableop_resource: 
identityИв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpь
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_48_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mulЬ
IdentityIdentity$conv2d_48/kernel/Regularizer/mul:z:03^conv2d_48/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp
Ю
Б
F__inference_conv2d_52_layer_call_and_return_conditional_losses_2144745

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%А2

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
х
G
+__inference_lambda_17_layer_call_fn_2148699

inputs
identity╤
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
GPU2 *0J 8В *O
fJRH
F__inference_lambda_17_layer_call_and_return_conditional_losses_21446812
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
х
G
+__inference_lambda_17_layer_call_fn_2148704

inputs
identity╤
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
GPU2 *0J 8В *O
fJRH
F__inference_lambda_17_layer_call_and_return_conditional_losses_21450802
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ё
И
/__inference_sequential_16_layer_call_fn_2147365
lambda_16_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCalllambda_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_21443122
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         KK
)
_user_specified_namelambda_16_input
╢
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_2144045

inputs
identityИc
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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧г
й!
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2146618

inputsJ
<sequential_16_batch_normalization_16_readvariableop_resource:L
>sequential_16_batch_normalization_16_readvariableop_1_resource:[
Msequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:]
Osequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_16_conv2d_48_conv2d_readvariableop_resource: E
7sequential_16_conv2d_48_biasadd_readvariableop_resource: Q
6sequential_16_conv2d_49_conv2d_readvariableop_resource: АF
7sequential_16_conv2d_49_biasadd_readvariableop_resource:	АR
6sequential_16_conv2d_50_conv2d_readvariableop_resource:ААF
7sequential_16_conv2d_50_biasadd_readvariableop_resource:	АJ
5sequential_16_dense_40_matmul_readvariableop_resource:АвАE
6sequential_16_dense_40_biasadd_readvariableop_resource:	АI
5sequential_16_dense_41_matmul_readvariableop_resource:
ААE
6sequential_16_dense_41_biasadd_readvariableop_resource:	АJ
<sequential_17_batch_normalization_17_readvariableop_resource:L
>sequential_17_batch_normalization_17_readvariableop_1_resource:[
Msequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_17_conv2d_51_conv2d_readvariableop_resource: E
7sequential_17_conv2d_51_biasadd_readvariableop_resource: Q
6sequential_17_conv2d_52_conv2d_readvariableop_resource: АF
7sequential_17_conv2d_52_biasadd_readvariableop_resource:	АR
6sequential_17_conv2d_53_conv2d_readvariableop_resource:ААF
7sequential_17_conv2d_53_biasadd_readvariableop_resource:	АJ
5sequential_17_dense_42_matmul_readvariableop_resource:АвАE
6sequential_17_dense_42_biasadd_readvariableop_resource:	АI
5sequential_17_dense_43_matmul_readvariableop_resource:
ААE
6sequential_17_dense_43_biasadd_readvariableop_resource:	А:
'dense_44_matmul_readvariableop_resource:	А6
(dense_44_biasadd_readvariableop_resource:
identityИв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв2conv2d_51/kernel/Regularizer/Square/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpв1dense_42/kernel/Regularizer/Square/ReadVariableOpв1dense_43/kernel/Regularizer/Square/ReadVariableOpвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpвDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpвFsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в3sequential_16/batch_normalization_16/ReadVariableOpв5sequential_16/batch_normalization_16/ReadVariableOp_1в.sequential_16/conv2d_48/BiasAdd/ReadVariableOpв-sequential_16/conv2d_48/Conv2D/ReadVariableOpв.sequential_16/conv2d_49/BiasAdd/ReadVariableOpв-sequential_16/conv2d_49/Conv2D/ReadVariableOpв.sequential_16/conv2d_50/BiasAdd/ReadVariableOpв-sequential_16/conv2d_50/Conv2D/ReadVariableOpв-sequential_16/dense_40/BiasAdd/ReadVariableOpв,sequential_16/dense_40/MatMul/ReadVariableOpв-sequential_16/dense_41/BiasAdd/ReadVariableOpв,sequential_16/dense_41/MatMul/ReadVariableOpвDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpвFsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в3sequential_17/batch_normalization_17/ReadVariableOpв5sequential_17/batch_normalization_17/ReadVariableOp_1в.sequential_17/conv2d_51/BiasAdd/ReadVariableOpв-sequential_17/conv2d_51/Conv2D/ReadVariableOpв.sequential_17/conv2d_52/BiasAdd/ReadVariableOpв-sequential_17/conv2d_52/Conv2D/ReadVariableOpв.sequential_17/conv2d_53/BiasAdd/ReadVariableOpв-sequential_17/conv2d_53/Conv2D/ReadVariableOpв-sequential_17/dense_42/BiasAdd/ReadVariableOpв,sequential_17/dense_42/MatMul/ReadVariableOpв-sequential_17/dense_43/BiasAdd/ReadVariableOpв,sequential_17/dense_43/MatMul/ReadVariableOp│
+sequential_16/lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_16/lambda_16/strided_slice/stack╖
-sequential_16/lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2/
-sequential_16/lambda_16/strided_slice/stack_1╖
-sequential_16/lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_16/lambda_16/strided_slice/stack_2ї
%sequential_16/lambda_16/strided_sliceStridedSliceinputs4sequential_16/lambda_16/strided_slice/stack:output:06sequential_16/lambda_16/strided_slice/stack_1:output:06sequential_16/lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_16/lambda_16/strided_sliceу
3sequential_16/batch_normalization_16/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_16/batch_normalization_16/ReadVariableOpщ
5sequential_16/batch_normalization_16/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_16/batch_normalization_16/ReadVariableOp_1Ц
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1╨
5sequential_16/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3.sequential_16/lambda_16/strided_slice:output:0;sequential_16/batch_normalization_16/ReadVariableOp:value:0=sequential_16/batch_normalization_16/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 27
5sequential_16/batch_normalization_16/FusedBatchNormV3▌
-sequential_16/conv2d_48/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_48/Conv2D/ReadVariableOpЮ
sequential_16/conv2d_48/Conv2DConv2D9sequential_16/batch_normalization_16/FusedBatchNormV3:y:05sequential_16/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_16/conv2d_48/Conv2D╘
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpш
sequential_16/conv2d_48/BiasAddBiasAdd'sequential_16/conv2d_48/Conv2D:output:06sequential_16/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_16/conv2d_48/BiasAddи
sequential_16/conv2d_48/ReluRelu(sequential_16/conv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_16/conv2d_48/ReluЇ
&sequential_16/max_pooling2d_48/MaxPoolMaxPool*sequential_16/conv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool▐
-sequential_16/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_16/conv2d_49/Conv2D/ReadVariableOpХ
sequential_16/conv2d_49/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_16/conv2d_49/Conv2D╒
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_49/BiasAddBiasAdd'sequential_16/conv2d_49/Conv2D:output:06sequential_16/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_16/conv2d_49/BiasAddй
sequential_16/conv2d_49/ReluRelu(sequential_16/conv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_16/conv2d_49/Reluї
&sequential_16/max_pooling2d_49/MaxPoolMaxPool*sequential_16/conv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool▀
-sequential_16/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_16/conv2d_50/Conv2D/ReadVariableOpХ
sequential_16/conv2d_50/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:05sequential_16/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_16/conv2d_50/Conv2D╒
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_50/BiasAddBiasAdd'sequential_16/conv2d_50/Conv2D:output:06sequential_16/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_16/conv2d_50/BiasAddй
sequential_16/conv2d_50/ReluRelu(sequential_16/conv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_16/conv2d_50/Reluї
&sequential_16/max_pooling2d_50/MaxPoolMaxPool*sequential_16/conv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPool╛
!sequential_16/dropout_48/IdentityIdentity/sequential_16/max_pooling2d_50/MaxPool:output:0*
T0*0
_output_shapes
:         		А2#
!sequential_16/dropout_48/IdentityС
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_16/flatten_16/Const╪
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_48/Identity:output:0'sequential_16/flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_16/flatten_16/Reshape╒
,sequential_16/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_16/dense_40/MatMul/ReadVariableOp▄
sequential_16/dense_40/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/MatMul╥
-sequential_16/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_40/BiasAdd/ReadVariableOp▐
sequential_16/dense_40/BiasAddBiasAdd'sequential_16/dense_40/MatMul:product:05sequential_16/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_40/BiasAddЮ
sequential_16/dense_40/ReluRelu'sequential_16/dense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/Relu░
!sequential_16/dropout_49/IdentityIdentity)sequential_16/dense_40/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_16/dropout_49/Identity╘
,sequential_16/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_16/dense_41/MatMul/ReadVariableOp▌
sequential_16/dense_41/MatMulMatMul*sequential_16/dropout_49/Identity:output:04sequential_16/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/MatMul╥
-sequential_16/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_41/BiasAdd/ReadVariableOp▐
sequential_16/dense_41/BiasAddBiasAdd'sequential_16/dense_41/MatMul:product:05sequential_16/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_41/BiasAddЮ
sequential_16/dense_41/ReluRelu'sequential_16/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/Relu░
!sequential_16/dropout_50/IdentityIdentity)sequential_16/dense_41/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_16/dropout_50/Identity│
+sequential_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2-
+sequential_17/lambda_17/strided_slice/stack╖
-sequential_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2/
-sequential_17/lambda_17/strided_slice/stack_1╖
-sequential_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_17/lambda_17/strided_slice/stack_2ї
%sequential_17/lambda_17/strided_sliceStridedSliceinputs4sequential_17/lambda_17/strided_slice/stack:output:06sequential_17/lambda_17/strided_slice/stack_1:output:06sequential_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_17/lambda_17/strided_sliceу
3sequential_17/batch_normalization_17/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_17/batch_normalization_17/ReadVariableOpщ
5sequential_17/batch_normalization_17/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_17/batch_normalization_17/ReadVariableOp_1Ц
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1╨
5sequential_17/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3.sequential_17/lambda_17/strided_slice:output:0;sequential_17/batch_normalization_17/ReadVariableOp:value:0=sequential_17/batch_normalization_17/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 27
5sequential_17/batch_normalization_17/FusedBatchNormV3▌
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOpЮ
sequential_17/conv2d_51/Conv2DConv2D9sequential_17/batch_normalization_17/FusedBatchNormV3:y:05sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_17/conv2d_51/Conv2D╘
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpш
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_17/conv2d_51/BiasAddи
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_17/conv2d_51/ReluЇ
&sequential_17/max_pooling2d_51/MaxPoolMaxPool*sequential_17/conv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_51/MaxPool▐
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOpХ
sequential_17/conv2d_52/Conv2DConv2D/sequential_17/max_pooling2d_51/MaxPool:output:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_17/conv2d_52/Conv2D╒
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_17/conv2d_52/BiasAddй
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_17/conv2d_52/Reluї
&sequential_17/max_pooling2d_52/MaxPoolMaxPool*sequential_17/conv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_52/MaxPool▀
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOpХ
sequential_17/conv2d_53/Conv2DConv2D/sequential_17/max_pooling2d_52/MaxPool:output:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_17/conv2d_53/Conv2D╒
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_17/conv2d_53/BiasAddй
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_17/conv2d_53/Reluї
&sequential_17/max_pooling2d_53/MaxPoolMaxPool*sequential_17/conv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_53/MaxPool╛
!sequential_17/dropout_51/IdentityIdentity/sequential_17/max_pooling2d_53/MaxPool:output:0*
T0*0
_output_shapes
:         		А2#
!sequential_17/dropout_51/IdentityС
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_17/flatten_17/Const╪
 sequential_17/flatten_17/ReshapeReshape*sequential_17/dropout_51/Identity:output:0'sequential_17/flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_17/flatten_17/Reshape╒
,sequential_17/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_17/dense_42/MatMul/ReadVariableOp▄
sequential_17/dense_42/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/MatMul╥
-sequential_17/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_42/BiasAdd/ReadVariableOp▐
sequential_17/dense_42/BiasAddBiasAdd'sequential_17/dense_42/MatMul:product:05sequential_17/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_42/BiasAddЮ
sequential_17/dense_42/ReluRelu'sequential_17/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/Relu░
!sequential_17/dropout_52/IdentityIdentity)sequential_17/dense_42/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_17/dropout_52/Identity╘
,sequential_17/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_17/dense_43/MatMul/ReadVariableOp▌
sequential_17/dense_43/MatMulMatMul*sequential_17/dropout_52/Identity:output:04sequential_17/dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/MatMul╥
-sequential_17/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_43/BiasAdd/ReadVariableOp▐
sequential_17/dense_43/BiasAddBiasAdd'sequential_17/dense_43/MatMul:product:05sequential_17/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_43/BiasAddЮ
sequential_17/dense_43/ReluRelu'sequential_17/dense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/Relu░
!sequential_17/dropout_53/IdentityIdentity)sequential_17/dense_43/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_17/dropout_53/Identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╞
concatConcatV2*sequential_16/dropout_50/Identity:output:0*sequential_17/dropout_53/Identity:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatй
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_44/MatMul/ReadVariableOpЧ
dense_44/MatMulMatMulconcat:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpе
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/BiasAdd|
dense_44/SoftmaxSoftmaxdense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_44/Softmaxч
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul▀
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul▐
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulч
2conv2d_51/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_51/kernel/Regularizer/SquareSquare:conv2d_51/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_51/kernel/Regularizer/Squareб
"conv2d_51/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_51/kernel/Regularizer/Const┬
 conv2d_51/kernel/Regularizer/SumSum'conv2d_51/kernel/Regularizer/Square:y:0+conv2d_51/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/SumН
"conv2d_51/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_51/kernel/Regularizer/mul/x─
 conv2d_51/kernel/Regularizer/mulMul+conv2d_51/kernel/Regularizer/mul/x:output:0)conv2d_51/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_51/kernel/Regularizer/mul▀
1dense_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_42/kernel/Regularizer/Square/ReadVariableOp╣
"dense_42/kernel/Regularizer/SquareSquare9dense_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_42/kernel/Regularizer/SquareЧ
!dense_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_42/kernel/Regularizer/Const╛
dense_42/kernel/Regularizer/SumSum&dense_42/kernel/Regularizer/Square:y:0*dense_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/SumЛ
!dense_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_42/kernel/Regularizer/mul/x└
dense_42/kernel/Regularizer/mulMul*dense_42/kernel/Regularizer/mul/x:output:0(dense_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_42/kernel/Regularizer/mul▐
1dense_43/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_43/kernel/Regularizer/Square/ReadVariableOp╕
"dense_43/kernel/Regularizer/SquareSquare9dense_43/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_43/kernel/Regularizer/SquareЧ
!dense_43/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_43/kernel/Regularizer/Const╛
dense_43/kernel/Regularizer/SumSum&dense_43/kernel/Regularizer/Square:y:0*dense_43/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/SumЛ
!dense_43/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_43/kernel/Regularizer/mul/x└
dense_43/kernel/Regularizer/mulMul*dense_43/kernel/Regularizer/mul/x:output:0(dense_43/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_43/kernel/Regularizer/mulй
IdentityIdentitydense_44/Softmax:softmax:03^conv2d_48/kernel/Regularizer/Square/ReadVariableOp3^conv2d_51/kernel/Regularizer/Square/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp2^dense_42/kernel/Regularizer/Square/ReadVariableOp2^dense_43/kernel/Regularizer/Square/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOpE^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_16/ReadVariableOp6^sequential_16/batch_normalization_16/ReadVariableOp_1/^sequential_16/conv2d_48/BiasAdd/ReadVariableOp.^sequential_16/conv2d_48/Conv2D/ReadVariableOp/^sequential_16/conv2d_49/BiasAdd/ReadVariableOp.^sequential_16/conv2d_49/Conv2D/ReadVariableOp/^sequential_16/conv2d_50/BiasAdd/ReadVariableOp.^sequential_16/conv2d_50/Conv2D/ReadVariableOp.^sequential_16/dense_40/BiasAdd/ReadVariableOp-^sequential_16/dense_40/MatMul/ReadVariableOp.^sequential_16/dense_41/BiasAdd/ReadVariableOp-^sequential_16/dense_41/MatMul/ReadVariableOpE^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_17/ReadVariableOp6^sequential_17/batch_normalization_17/ReadVariableOp_1/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_42/BiasAdd/ReadVariableOp-^sequential_17/dense_42/MatMul/ReadVariableOp.^sequential_17/dense_43/BiasAdd/ReadVariableOp-^sequential_17/dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2conv2d_51/kernel/Regularizer/Square/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2f
1dense_42/kernel/Regularizer/Square/ReadVariableOp1dense_42/kernel/Regularizer/Square/ReadVariableOp2f
1dense_43/kernel/Regularizer/Square/ReadVariableOp1dense_43/kernel/Regularizer/Square/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2М
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_16/ReadVariableOp3sequential_16/batch_normalization_16/ReadVariableOp2n
5sequential_16/batch_normalization_16/ReadVariableOp_15sequential_16/batch_normalization_16/ReadVariableOp_12`
.sequential_16/conv2d_48/BiasAdd/ReadVariableOp.sequential_16/conv2d_48/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_48/Conv2D/ReadVariableOp-sequential_16/conv2d_48/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_49/BiasAdd/ReadVariableOp.sequential_16/conv2d_49/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_49/Conv2D/ReadVariableOp-sequential_16/conv2d_49/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_50/BiasAdd/ReadVariableOp.sequential_16/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_50/Conv2D/ReadVariableOp-sequential_16/conv2d_50/Conv2D/ReadVariableOp2^
-sequential_16/dense_40/BiasAdd/ReadVariableOp-sequential_16/dense_40/BiasAdd/ReadVariableOp2\
,sequential_16/dense_40/MatMul/ReadVariableOp,sequential_16/dense_40/MatMul/ReadVariableOp2^
-sequential_16/dense_41/BiasAdd/ReadVariableOp-sequential_16/dense_41/BiasAdd/ReadVariableOp2\
,sequential_16/dense_41/MatMul/ReadVariableOp,sequential_16/dense_41/MatMul/ReadVariableOp2М
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_17/ReadVariableOp3sequential_17/batch_normalization_17/ReadVariableOp2n
5sequential_17/batch_normalization_17/ReadVariableOp_15sequential_17/batch_normalization_17/ReadVariableOp_12`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_42/BiasAdd/ReadVariableOp-sequential_17/dense_42/BiasAdd/ReadVariableOp2\
,sequential_17/dense_42/MatMul/ReadVariableOp,sequential_17/dense_42/MatMul/ReadVariableOp2^
-sequential_17/dense_43/BiasAdd/ReadVariableOp-sequential_17/dense_43/BiasAdd/ReadVariableOp2\
,sequential_17/dense_43/MatMul/ReadVariableOp,sequential_17/dense_43/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
к
╙
8__inference_batch_normalization_16_layer_call_fn_2148348

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallк
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_21438302
StatefulPartitionedCallЦ
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
─
b
F__inference_lambda_16_layer_call_and_return_conditional_losses_2143811

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┼
Ю
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148415

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
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
epsilon%oГ:*
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
║х
┬
__inference_call_1989106

inputsJ
<sequential_16_batch_normalization_16_readvariableop_resource:L
>sequential_16_batch_normalization_16_readvariableop_1_resource:[
Msequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:]
Osequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_16_conv2d_48_conv2d_readvariableop_resource: E
7sequential_16_conv2d_48_biasadd_readvariableop_resource: Q
6sequential_16_conv2d_49_conv2d_readvariableop_resource: АF
7sequential_16_conv2d_49_biasadd_readvariableop_resource:	АR
6sequential_16_conv2d_50_conv2d_readvariableop_resource:ААF
7sequential_16_conv2d_50_biasadd_readvariableop_resource:	АJ
5sequential_16_dense_40_matmul_readvariableop_resource:АвАE
6sequential_16_dense_40_biasadd_readvariableop_resource:	АI
5sequential_16_dense_41_matmul_readvariableop_resource:
ААE
6sequential_16_dense_41_biasadd_readvariableop_resource:	АJ
<sequential_17_batch_normalization_17_readvariableop_resource:L
>sequential_17_batch_normalization_17_readvariableop_1_resource:[
Msequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_17_conv2d_51_conv2d_readvariableop_resource: E
7sequential_17_conv2d_51_biasadd_readvariableop_resource: Q
6sequential_17_conv2d_52_conv2d_readvariableop_resource: АF
7sequential_17_conv2d_52_biasadd_readvariableop_resource:	АR
6sequential_17_conv2d_53_conv2d_readvariableop_resource:ААF
7sequential_17_conv2d_53_biasadd_readvariableop_resource:	АJ
5sequential_17_dense_42_matmul_readvariableop_resource:АвАE
6sequential_17_dense_42_biasadd_readvariableop_resource:	АI
5sequential_17_dense_43_matmul_readvariableop_resource:
ААE
6sequential_17_dense_43_biasadd_readvariableop_resource:	А:
'dense_44_matmul_readvariableop_resource:	А6
(dense_44_biasadd_readvariableop_resource:
identityИвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpвDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpвFsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в3sequential_16/batch_normalization_16/ReadVariableOpв5sequential_16/batch_normalization_16/ReadVariableOp_1в.sequential_16/conv2d_48/BiasAdd/ReadVariableOpв-sequential_16/conv2d_48/Conv2D/ReadVariableOpв.sequential_16/conv2d_49/BiasAdd/ReadVariableOpв-sequential_16/conv2d_49/Conv2D/ReadVariableOpв.sequential_16/conv2d_50/BiasAdd/ReadVariableOpв-sequential_16/conv2d_50/Conv2D/ReadVariableOpв-sequential_16/dense_40/BiasAdd/ReadVariableOpв,sequential_16/dense_40/MatMul/ReadVariableOpв-sequential_16/dense_41/BiasAdd/ReadVariableOpв,sequential_16/dense_41/MatMul/ReadVariableOpвDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpвFsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в3sequential_17/batch_normalization_17/ReadVariableOpв5sequential_17/batch_normalization_17/ReadVariableOp_1в.sequential_17/conv2d_51/BiasAdd/ReadVariableOpв-sequential_17/conv2d_51/Conv2D/ReadVariableOpв.sequential_17/conv2d_52/BiasAdd/ReadVariableOpв-sequential_17/conv2d_52/Conv2D/ReadVariableOpв.sequential_17/conv2d_53/BiasAdd/ReadVariableOpв-sequential_17/conv2d_53/Conv2D/ReadVariableOpв-sequential_17/dense_42/BiasAdd/ReadVariableOpв,sequential_17/dense_42/MatMul/ReadVariableOpв-sequential_17/dense_43/BiasAdd/ReadVariableOpв,sequential_17/dense_43/MatMul/ReadVariableOp│
+sequential_16/lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_16/lambda_16/strided_slice/stack╖
-sequential_16/lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2/
-sequential_16/lambda_16/strided_slice/stack_1╖
-sequential_16/lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_16/lambda_16/strided_slice/stack_2ї
%sequential_16/lambda_16/strided_sliceStridedSliceinputs4sequential_16/lambda_16/strided_slice/stack:output:06sequential_16/lambda_16/strided_slice/stack_1:output:06sequential_16/lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_16/lambda_16/strided_sliceу
3sequential_16/batch_normalization_16/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_16/batch_normalization_16/ReadVariableOpщ
5sequential_16/batch_normalization_16/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_16/batch_normalization_16/ReadVariableOp_1Ц
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1╨
5sequential_16/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3.sequential_16/lambda_16/strided_slice:output:0;sequential_16/batch_normalization_16/ReadVariableOp:value:0=sequential_16/batch_normalization_16/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 27
5sequential_16/batch_normalization_16/FusedBatchNormV3▌
-sequential_16/conv2d_48/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_48/Conv2D/ReadVariableOpЮ
sequential_16/conv2d_48/Conv2DConv2D9sequential_16/batch_normalization_16/FusedBatchNormV3:y:05sequential_16/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_16/conv2d_48/Conv2D╘
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_48/BiasAdd/ReadVariableOpш
sequential_16/conv2d_48/BiasAddBiasAdd'sequential_16/conv2d_48/Conv2D:output:06sequential_16/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_16/conv2d_48/BiasAddи
sequential_16/conv2d_48/ReluRelu(sequential_16/conv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_16/conv2d_48/ReluЇ
&sequential_16/max_pooling2d_48/MaxPoolMaxPool*sequential_16/conv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool▐
-sequential_16/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_16/conv2d_49/Conv2D/ReadVariableOpХ
sequential_16/conv2d_49/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_16/conv2d_49/Conv2D╒
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_49/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_49/BiasAddBiasAdd'sequential_16/conv2d_49/Conv2D:output:06sequential_16/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_16/conv2d_49/BiasAddй
sequential_16/conv2d_49/ReluRelu(sequential_16/conv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_16/conv2d_49/Reluї
&sequential_16/max_pooling2d_49/MaxPoolMaxPool*sequential_16/conv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool▀
-sequential_16/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_16/conv2d_50/Conv2D/ReadVariableOpХ
sequential_16/conv2d_50/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:05sequential_16/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_16/conv2d_50/Conv2D╒
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_16/conv2d_50/BiasAdd/ReadVariableOpщ
sequential_16/conv2d_50/BiasAddBiasAdd'sequential_16/conv2d_50/Conv2D:output:06sequential_16/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_16/conv2d_50/BiasAddй
sequential_16/conv2d_50/ReluRelu(sequential_16/conv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_16/conv2d_50/Reluї
&sequential_16/max_pooling2d_50/MaxPoolMaxPool*sequential_16/conv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPool╛
!sequential_16/dropout_48/IdentityIdentity/sequential_16/max_pooling2d_50/MaxPool:output:0*
T0*0
_output_shapes
:         		А2#
!sequential_16/dropout_48/IdentityС
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_16/flatten_16/Const╪
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_48/Identity:output:0'sequential_16/flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_16/flatten_16/Reshape╒
,sequential_16/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_16/dense_40/MatMul/ReadVariableOp▄
sequential_16/dense_40/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/MatMul╥
-sequential_16/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_40/BiasAdd/ReadVariableOp▐
sequential_16/dense_40/BiasAddBiasAdd'sequential_16/dense_40/MatMul:product:05sequential_16/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_40/BiasAddЮ
sequential_16/dense_40/ReluRelu'sequential_16/dense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_40/Relu░
!sequential_16/dropout_49/IdentityIdentity)sequential_16/dense_40/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_16/dropout_49/Identity╘
,sequential_16/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_16/dense_41/MatMul/ReadVariableOp▌
sequential_16/dense_41/MatMulMatMul*sequential_16/dropout_49/Identity:output:04sequential_16/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/MatMul╥
-sequential_16/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_16/dense_41/BiasAdd/ReadVariableOp▐
sequential_16/dense_41/BiasAddBiasAdd'sequential_16/dense_41/MatMul:product:05sequential_16/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_16/dense_41/BiasAddЮ
sequential_16/dense_41/ReluRelu'sequential_16/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_16/dense_41/Relu░
!sequential_16/dropout_50/IdentityIdentity)sequential_16/dense_41/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_16/dropout_50/Identity│
+sequential_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2-
+sequential_17/lambda_17/strided_slice/stack╖
-sequential_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2/
-sequential_17/lambda_17/strided_slice/stack_1╖
-sequential_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2/
-sequential_17/lambda_17/strided_slice/stack_2ї
%sequential_17/lambda_17/strided_sliceStridedSliceinputs4sequential_17/lambda_17/strided_slice/stack:output:06sequential_17/lambda_17/strided_slice/stack_1:output:06sequential_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2'
%sequential_17/lambda_17/strided_sliceу
3sequential_17/batch_normalization_17/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_17_readvariableop_resource*
_output_shapes
:*
dtype025
3sequential_17/batch_normalization_17/ReadVariableOpщ
5sequential_17/batch_normalization_17/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:*
dtype027
5sequential_17/batch_normalization_17/ReadVariableOp_1Ц
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1╨
5sequential_17/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3.sequential_17/lambda_17/strided_slice:output:0;sequential_17/batch_normalization_17/ReadVariableOp:value:0=sequential_17/batch_normalization_17/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 27
5sequential_17/batch_normalization_17/FusedBatchNormV3▌
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOpЮ
sequential_17/conv2d_51/Conv2DConv2D9sequential_17/batch_normalization_17/FusedBatchNormV3:y:05sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2 
sequential_17/conv2d_51/Conv2D╘
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpш
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2!
sequential_17/conv2d_51/BiasAddи
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_17/conv2d_51/ReluЇ
&sequential_17/max_pooling2d_51/MaxPoolMaxPool*sequential_17/conv2d_51/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_51/MaxPool▐
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOpХ
sequential_17/conv2d_52/Conv2DConv2D/sequential_17/max_pooling2d_51/MaxPool:output:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2 
sequential_17/conv2d_52/Conv2D╒
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2!
sequential_17/conv2d_52/BiasAddй
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_17/conv2d_52/Reluї
&sequential_17/max_pooling2d_52/MaxPoolMaxPool*sequential_17/conv2d_52/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_52/MaxPool▀
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOpХ
sequential_17/conv2d_53/Conv2DConv2D/sequential_17/max_pooling2d_52/MaxPool:output:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_17/conv2d_53/Conv2D╒
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpщ
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_17/conv2d_53/BiasAddй
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_17/conv2d_53/Reluї
&sequential_17/max_pooling2d_53/MaxPoolMaxPool*sequential_17/conv2d_53/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_53/MaxPool╛
!sequential_17/dropout_51/IdentityIdentity/sequential_17/max_pooling2d_53/MaxPool:output:0*
T0*0
_output_shapes
:         		А2#
!sequential_17/dropout_51/IdentityС
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2 
sequential_17/flatten_17/Const╪
 sequential_17/flatten_17/ReshapeReshape*sequential_17/dropout_51/Identity:output:0'sequential_17/flatten_17/Const:output:0*
T0*)
_output_shapes
:         Ав2"
 sequential_17/flatten_17/Reshape╒
,sequential_17/dense_42/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_42_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02.
,sequential_17/dense_42/MatMul/ReadVariableOp▄
sequential_17/dense_42/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/MatMul╥
-sequential_17/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_42/BiasAdd/ReadVariableOp▐
sequential_17/dense_42/BiasAddBiasAdd'sequential_17/dense_42/MatMul:product:05sequential_17/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_42/BiasAddЮ
sequential_17/dense_42/ReluRelu'sequential_17/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_42/Relu░
!sequential_17/dropout_52/IdentityIdentity)sequential_17/dense_42/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_17/dropout_52/Identity╘
,sequential_17/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential_17/dense_43/MatMul/ReadVariableOp▌
sequential_17/dense_43/MatMulMatMul*sequential_17/dropout_52/Identity:output:04sequential_17/dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/MatMul╥
-sequential_17/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_17/dense_43/BiasAdd/ReadVariableOp▐
sequential_17/dense_43/BiasAddBiasAdd'sequential_17/dense_43/MatMul:product:05sequential_17/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_17/dense_43/BiasAddЮ
sequential_17/dense_43/ReluRelu'sequential_17/dense_43/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_17/dense_43/Relu░
!sequential_17/dropout_53/IdentityIdentity)sequential_17/dense_43/Relu:activations:0*
T0*(
_output_shapes
:         А2#
!sequential_17/dropout_53/Identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╞
concatConcatV2*sequential_16/dropout_50/Identity:output:0*sequential_17/dropout_53/Identity:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatй
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_44/MatMul/ReadVariableOpЧ
dense_44/MatMulMatMulconcat:output:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpе
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/BiasAdd|
dense_44/SoftmaxSoftmaxdense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_44/Softmaxя
IdentityIdentitydense_44/Softmax:softmax:0 ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOpE^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_16/ReadVariableOp6^sequential_16/batch_normalization_16/ReadVariableOp_1/^sequential_16/conv2d_48/BiasAdd/ReadVariableOp.^sequential_16/conv2d_48/Conv2D/ReadVariableOp/^sequential_16/conv2d_49/BiasAdd/ReadVariableOp.^sequential_16/conv2d_49/Conv2D/ReadVariableOp/^sequential_16/conv2d_50/BiasAdd/ReadVariableOp.^sequential_16/conv2d_50/Conv2D/ReadVariableOp.^sequential_16/dense_40/BiasAdd/ReadVariableOp-^sequential_16/dense_40/MatMul/ReadVariableOp.^sequential_16/dense_41/BiasAdd/ReadVariableOp-^sequential_16/dense_41/MatMul/ReadVariableOpE^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_17/ReadVariableOp6^sequential_17/batch_normalization_17/ReadVariableOp_1/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_42/BiasAdd/ReadVariableOp-^sequential_17/dense_42/MatMul/ReadVariableOp.^sequential_17/dense_43/BiasAdd/ReadVariableOp-^sequential_17/dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2М
Dsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_16/ReadVariableOp3sequential_16/batch_normalization_16/ReadVariableOp2n
5sequential_16/batch_normalization_16/ReadVariableOp_15sequential_16/batch_normalization_16/ReadVariableOp_12`
.sequential_16/conv2d_48/BiasAdd/ReadVariableOp.sequential_16/conv2d_48/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_48/Conv2D/ReadVariableOp-sequential_16/conv2d_48/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_49/BiasAdd/ReadVariableOp.sequential_16/conv2d_49/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_49/Conv2D/ReadVariableOp-sequential_16/conv2d_49/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_50/BiasAdd/ReadVariableOp.sequential_16/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_50/Conv2D/ReadVariableOp-sequential_16/conv2d_50/Conv2D/ReadVariableOp2^
-sequential_16/dense_40/BiasAdd/ReadVariableOp-sequential_16/dense_40/BiasAdd/ReadVariableOp2\
,sequential_16/dense_40/MatMul/ReadVariableOp,sequential_16/dense_40/MatMul/ReadVariableOp2^
-sequential_16/dense_41/BiasAdd/ReadVariableOp-sequential_16/dense_41/BiasAdd/ReadVariableOp2\
,sequential_16/dense_41/MatMul/ReadVariableOp,sequential_16/dense_41/MatMul/ReadVariableOp2М
Dsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_17/ReadVariableOp3sequential_17/batch_normalization_17/ReadVariableOp2n
5sequential_17/batch_normalization_17/ReadVariableOp_15sequential_17/batch_normalization_17/ReadVariableOp_12`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_42/BiasAdd/ReadVariableOp-sequential_17/dense_42/BiasAdd/ReadVariableOp2\
,sequential_17/dense_42/MatMul/ReadVariableOp,sequential_17/dense_42/MatMul/ReadVariableOp2^
-sequential_17/dense_43/BiasAdd/ReadVariableOp-sequential_17/dense_43/BiasAdd/ReadVariableOp2\
,sequential_17/dense_43/MatMul/ReadVariableOp,sequential_17/dense_43/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ш
e
G__inference_dropout_51_layer_call_and_return_conditional_losses_2144775

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         		А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╓и
Б'
 __inference__traced_save_2149401
file_prefix.
*savev2_dense_44_kernel_read_readvariableop,
(savev2_dense_44_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop/
+savev2_conv2d_48_kernel_read_readvariableop-
)savev2_conv2d_48_bias_read_readvariableop/
+savev2_conv2d_49_kernel_read_readvariableop-
)savev2_conv2d_49_bias_read_readvariableop/
+savev2_conv2d_50_kernel_read_readvariableop-
)savev2_conv2d_50_bias_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop/
+savev2_conv2d_52_kernel_read_readvariableop-
)savev2_conv2d_52_bias_read_readvariableop/
+savev2_conv2d_53_kernel_read_readvariableop-
)savev2_conv2d_53_bias_read_readvariableop.
*savev2_dense_42_kernel_read_readvariableop,
(savev2_dense_42_bias_read_readvariableop.
*savev2_dense_43_kernel_read_readvariableop,
(savev2_dense_43_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_44_kernel_m_read_readvariableop3
/savev2_adam_dense_44_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_m_read_readvariableop6
2savev2_adam_conv2d_48_kernel_m_read_readvariableop4
0savev2_adam_conv2d_48_bias_m_read_readvariableop6
2savev2_adam_conv2d_49_kernel_m_read_readvariableop4
0savev2_adam_conv2d_49_bias_m_read_readvariableop6
2savev2_adam_conv2d_50_kernel_m_read_readvariableop4
0savev2_adam_conv2d_50_bias_m_read_readvariableop5
1savev2_adam_dense_40_kernel_m_read_readvariableop3
/savev2_adam_dense_40_bias_m_read_readvariableop5
1savev2_adam_dense_41_kernel_m_read_readvariableop3
/savev2_adam_dense_41_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_m_read_readvariableop6
2savev2_adam_conv2d_51_kernel_m_read_readvariableop4
0savev2_adam_conv2d_51_bias_m_read_readvariableop6
2savev2_adam_conv2d_52_kernel_m_read_readvariableop4
0savev2_adam_conv2d_52_bias_m_read_readvariableop6
2savev2_adam_conv2d_53_kernel_m_read_readvariableop4
0savev2_adam_conv2d_53_bias_m_read_readvariableop5
1savev2_adam_dense_42_kernel_m_read_readvariableop3
/savev2_adam_dense_42_bias_m_read_readvariableop5
1savev2_adam_dense_43_kernel_m_read_readvariableop3
/savev2_adam_dense_43_bias_m_read_readvariableop5
1savev2_adam_dense_44_kernel_v_read_readvariableop3
/savev2_adam_dense_44_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_v_read_readvariableop6
2savev2_adam_conv2d_48_kernel_v_read_readvariableop4
0savev2_adam_conv2d_48_bias_v_read_readvariableop6
2savev2_adam_conv2d_49_kernel_v_read_readvariableop4
0savev2_adam_conv2d_49_bias_v_read_readvariableop6
2savev2_adam_conv2d_50_kernel_v_read_readvariableop4
0savev2_adam_conv2d_50_bias_v_read_readvariableop5
1savev2_adam_dense_40_kernel_v_read_readvariableop3
/savev2_adam_dense_40_bias_v_read_readvariableop5
1savev2_adam_dense_41_kernel_v_read_readvariableop3
/savev2_adam_dense_41_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_v_read_readvariableop6
2savev2_adam_conv2d_51_kernel_v_read_readvariableop4
0savev2_adam_conv2d_51_bias_v_read_readvariableop6
2savev2_adam_conv2d_52_kernel_v_read_readvariableop4
0savev2_adam_conv2d_52_bias_v_read_readvariableop6
2savev2_adam_conv2d_53_kernel_v_read_readvariableop4
0savev2_adam_conv2d_53_bias_v_read_readvariableop5
1savev2_adam_dense_42_kernel_v_read_readvariableop3
/savev2_adam_dense_42_bias_v_read_readvariableop5
1savev2_adam_dense_43_kernel_v_read_readvariableop3
/savev2_adam_dense_43_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЬ*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:\*
dtype0*о)
valueд)Bб)\B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names├
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:\*
dtype0*═
value├B└\B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╜%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_44_kernel_read_readvariableop(savev2_dense_44_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop+savev2_conv2d_48_kernel_read_readvariableop)savev2_conv2d_48_bias_read_readvariableop+savev2_conv2d_49_kernel_read_readvariableop)savev2_conv2d_49_bias_read_readvariableop+savev2_conv2d_50_kernel_read_readvariableop)savev2_conv2d_50_bias_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop+savev2_conv2d_52_kernel_read_readvariableop)savev2_conv2d_52_bias_read_readvariableop+savev2_conv2d_53_kernel_read_readvariableop)savev2_conv2d_53_bias_read_readvariableop*savev2_dense_42_kernel_read_readvariableop(savev2_dense_42_bias_read_readvariableop*savev2_dense_43_kernel_read_readvariableop(savev2_dense_43_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_44_kernel_m_read_readvariableop/savev2_adam_dense_44_bias_m_read_readvariableop>savev2_adam_batch_normalization_16_gamma_m_read_readvariableop=savev2_adam_batch_normalization_16_beta_m_read_readvariableop2savev2_adam_conv2d_48_kernel_m_read_readvariableop0savev2_adam_conv2d_48_bias_m_read_readvariableop2savev2_adam_conv2d_49_kernel_m_read_readvariableop0savev2_adam_conv2d_49_bias_m_read_readvariableop2savev2_adam_conv2d_50_kernel_m_read_readvariableop0savev2_adam_conv2d_50_bias_m_read_readvariableop1savev2_adam_dense_40_kernel_m_read_readvariableop/savev2_adam_dense_40_bias_m_read_readvariableop1savev2_adam_dense_41_kernel_m_read_readvariableop/savev2_adam_dense_41_bias_m_read_readvariableop>savev2_adam_batch_normalization_17_gamma_m_read_readvariableop=savev2_adam_batch_normalization_17_beta_m_read_readvariableop2savev2_adam_conv2d_51_kernel_m_read_readvariableop0savev2_adam_conv2d_51_bias_m_read_readvariableop2savev2_adam_conv2d_52_kernel_m_read_readvariableop0savev2_adam_conv2d_52_bias_m_read_readvariableop2savev2_adam_conv2d_53_kernel_m_read_readvariableop0savev2_adam_conv2d_53_bias_m_read_readvariableop1savev2_adam_dense_42_kernel_m_read_readvariableop/savev2_adam_dense_42_bias_m_read_readvariableop1savev2_adam_dense_43_kernel_m_read_readvariableop/savev2_adam_dense_43_bias_m_read_readvariableop1savev2_adam_dense_44_kernel_v_read_readvariableop/savev2_adam_dense_44_bias_v_read_readvariableop>savev2_adam_batch_normalization_16_gamma_v_read_readvariableop=savev2_adam_batch_normalization_16_beta_v_read_readvariableop2savev2_adam_conv2d_48_kernel_v_read_readvariableop0savev2_adam_conv2d_48_bias_v_read_readvariableop2savev2_adam_conv2d_49_kernel_v_read_readvariableop0savev2_adam_conv2d_49_bias_v_read_readvariableop2savev2_adam_conv2d_50_kernel_v_read_readvariableop0savev2_adam_conv2d_50_bias_v_read_readvariableop1savev2_adam_dense_40_kernel_v_read_readvariableop/savev2_adam_dense_40_bias_v_read_readvariableop1savev2_adam_dense_41_kernel_v_read_readvariableop/savev2_adam_dense_41_bias_v_read_readvariableop>savev2_adam_batch_normalization_17_gamma_v_read_readvariableop=savev2_adam_batch_normalization_17_beta_v_read_readvariableop2savev2_adam_conv2d_51_kernel_v_read_readvariableop0savev2_adam_conv2d_51_bias_v_read_readvariableop2savev2_adam_conv2d_52_kernel_v_read_readvariableop0savev2_adam_conv2d_52_bias_v_read_readvariableop2savev2_adam_conv2d_53_kernel_v_read_readvariableop0savev2_adam_conv2d_53_bias_v_read_readvariableop1savev2_adam_dense_42_kernel_v_read_readvariableop/savev2_adam_dense_42_bias_v_read_readvariableop1savev2_adam_dense_43_kernel_v_read_readvariableop/savev2_adam_dense_43_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *j
dtypes`
^2\	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*Ў
_input_shapesф
с: :	А:: : : : : ::::: : : А:А:АА:А:АвА:А:
АА:А::::: : : А:А:АА:А:АвА:А:
АА:А: : : : :	А:::: : : А:А:АА:А:АвА:А:
АА:А::: : : А:А:АА:А:АвА:А:
АА:А:	А:::: : : А:А:АА:А:АвА:А:
АА:А::: : : А:А:АА:А:АвА:А:
АА:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	А: 
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
: А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:'#
!
_output_shapes
:АвА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:' #
!
_output_shapes
:АвА:!!

_output_shapes	
:А:&""
 
_output_shapes
:
АА:!#

_output_shapes	
:А:$
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
:	А: )
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
: А:!/

_output_shapes	
:А:.0*
(
_output_shapes
:АА:!1

_output_shapes	
:А:'2#
!
_output_shapes
:АвА:!3

_output_shapes	
:А:&4"
 
_output_shapes
:
АА:!5

_output_shapes	
:А: 6

_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
: : 9

_output_shapes
: :-:)
'
_output_shapes
: А:!;

_output_shapes	
:А:.<*
(
_output_shapes
:АА:!=

_output_shapes	
:А:'>#
!
_output_shapes
:АвА:!?

_output_shapes	
:А:&@"
 
_output_shapes
:
АА:!A

_output_shapes	
:А:%B!

_output_shapes
:	А: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
: : G

_output_shapes
: :-H)
'
_output_shapes
: А:!I

_output_shapes	
:А:.J*
(
_output_shapes
:АА:!K

_output_shapes	
:А:'L#
!
_output_shapes
:АвА:!M

_output_shapes	
:А:&N"
 
_output_shapes
:
АА:!O

_output_shapes	
:А: P

_output_shapes
:: Q

_output_shapes
::,R(
&
_output_shapes
: : S

_output_shapes
: :-T)
'
_output_shapes
: А:!U

_output_shapes	
:А:.V*
(
_output_shapes
:АА:!W

_output_shapes	
:А:'X#
!
_output_shapes
:АвА:!Y

_output_shapes	
:А:&Z"
 
_output_shapes
:
АА:![

_output_shapes	
:А:\

_output_shapes
: 
Ш
e
G__inference_dropout_48_layer_call_and_return_conditional_losses_2143905

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         		А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╙
г
+__inference_conv2d_53_layer_call_fn_2148905

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_53_layer_call_and_return_conditional_losses_21447632
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
е
╫
*__inference_CNN_2jet_layer_call_fn_2146382

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: %

unknown_19: А

unknown_20:	А&

unknown_21:АА

unknown_22:	А

unknown_23:АвА

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCallї
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
:         *<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_21457402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:         KK: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╨
в
+__inference_conv2d_52_layer_call_fn_2148885

inputs"
unknown: А
	unknown_0:	А
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_52_layer_call_and_return_conditional_losses_21447452
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         %%А2

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
бЫ
Г
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147739
lambda_16_input<
.batch_normalization_16_readvariableop_resource:>
0batch_normalization_16_readvariableop_1_resource:M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_48_conv2d_readvariableop_resource: 7
)conv2d_48_biasadd_readvariableop_resource: C
(conv2d_49_conv2d_readvariableop_resource: А8
)conv2d_49_biasadd_readvariableop_resource:	АD
(conv2d_50_conv2d_readvariableop_resource:АА8
)conv2d_50_biasadd_readvariableop_resource:	А<
'dense_40_matmul_readvariableop_resource:АвА7
(dense_40_biasadd_readvariableop_resource:	А;
'dense_41_matmul_readvariableop_resource:
АА7
(dense_41_biasadd_readvariableop_resource:	А
identityИв%batch_normalization_16/AssignNewValueв'batch_normalization_16/AssignNewValue_1в6batch_normalization_16/FusedBatchNormV3/ReadVariableOpв8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_16/ReadVariableOpв'batch_normalization_16/ReadVariableOp_1в conv2d_48/BiasAdd/ReadVariableOpвconv2d_48/Conv2D/ReadVariableOpв2conv2d_48/kernel/Regularizer/Square/ReadVariableOpв conv2d_49/BiasAdd/ReadVariableOpвconv2d_49/Conv2D/ReadVariableOpв conv2d_50/BiasAdd/ReadVariableOpвconv2d_50/Conv2D/ReadVariableOpвdense_40/BiasAdd/ReadVariableOpвdense_40/MatMul/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpвdense_41/BiasAdd/ReadVariableOpвdense_41/MatMul/ReadVariableOpв1dense_41/kernel/Regularizer/Square/ReadVariableOpЧ
lambda_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_16/strided_slice/stackЫ
lambda_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2!
lambda_16/strided_slice/stack_1Ы
lambda_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2!
lambda_16/strided_slice/stack_2╕
lambda_16/strided_sliceStridedSlicelambda_16_input&lambda_16/strided_slice/stack:output:0(lambda_16/strided_slice/stack_1:output:0(lambda_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_16/strided_slice╣
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_16/ReadVariableOp┐
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_16/ReadVariableOp_1ь
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1№
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 lambda_16/strided_slice:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_16/FusedBatchNormV3╡
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValue┴
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1│
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_48/Conv2D/ReadVariableOpц
conv2d_48/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_48/Conv2Dк
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp░
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_48/BiasAdd~
conv2d_48/ReluReluconv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_48/Relu╩
max_pooling2d_48/MaxPoolMaxPoolconv2d_48/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_48/MaxPool┤
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_49/Conv2D/ReadVariableOp▌
conv2d_49/Conv2DConv2D!max_pooling2d_48/MaxPool:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_49/Conv2Dл
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp▒
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_49/BiasAdd
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_49/Relu╦
max_pooling2d_49/MaxPoolMaxPoolconv2d_49/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool╡
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_50/Conv2D/ReadVariableOp▌
conv2d_50/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_50/Conv2Dл
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp▒
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_50/BiasAdd
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_50/Relu╦
max_pooling2d_50/MaxPoolMaxPoolconv2d_50/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPooly
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_48/dropout/Const╕
dropout_48/dropout/MulMul!max_pooling2d_50/MaxPool:output:0!dropout_48/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout_48/dropout/MulЕ
dropout_48/dropout/ShapeShape!max_pooling2d_50/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_48/dropout/Shape▐
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype021
/dropout_48/dropout/random_uniform/RandomUniformЛ
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_48/dropout/GreaterEqual/yє
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2!
dropout_48/dropout/GreaterEqualй
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout_48/dropout/Castп
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout_48/dropout/Mul_1u
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_16/Constа
flatten_16/ReshapeReshapedropout_48/dropout/Mul_1:z:0flatten_16/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_16/Reshapeл
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02 
dense_40/MatMul/ReadVariableOpд
dense_40/MatMulMatMulflatten_16/Reshape:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_40/MatMulи
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_40/BiasAdd/ReadVariableOpж
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_40/BiasAddt
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_40/Reluy
dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_49/dropout/Constк
dropout_49/dropout/MulMuldense_40/Relu:activations:0!dropout_49/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_49/dropout/Mul
dropout_49/dropout/ShapeShapedense_40/Relu:activations:0*
T0*
_output_shapes
:2
dropout_49/dropout/Shape╓
/dropout_49/dropout/random_uniform/RandomUniformRandomUniform!dropout_49/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_49/dropout/random_uniform/RandomUniformЛ
!dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_49/dropout/GreaterEqual/yы
dropout_49/dropout/GreaterEqualGreaterEqual8dropout_49/dropout/random_uniform/RandomUniform:output:0*dropout_49/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_49/dropout/GreaterEqualб
dropout_49/dropout/CastCast#dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_49/dropout/Castз
dropout_49/dropout/Mul_1Muldropout_49/dropout/Mul:z:0dropout_49/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_49/dropout/Mul_1к
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_41/MatMul/ReadVariableOpе
dense_41/MatMulMatMuldropout_49/dropout/Mul_1:z:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_41/MatMulи
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_41/BiasAdd/ReadVariableOpж
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_41/BiasAddt
dense_41/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_41/Reluy
dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_50/dropout/Constк
dropout_50/dropout/MulMuldense_41/Relu:activations:0!dropout_50/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_50/dropout/Mul
dropout_50/dropout/ShapeShapedense_41/Relu:activations:0*
T0*
_output_shapes
:2
dropout_50/dropout/Shape╓
/dropout_50/dropout/random_uniform/RandomUniformRandomUniform!dropout_50/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_50/dropout/random_uniform/RandomUniformЛ
!dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_50/dropout/GreaterEqual/yы
dropout_50/dropout/GreaterEqualGreaterEqual8dropout_50/dropout/random_uniform/RandomUniform:output:0*dropout_50/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_50/dropout/GreaterEqualб
dropout_50/dropout/CastCast#dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_50/dropout/Castз
dropout_50/dropout/Mul_1Muldropout_50/dropout/Mul:z:0dropout_50/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_50/dropout/Mul_1┘
2conv2d_48/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_48/kernel/Regularizer/SquareSquare:conv2d_48/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_48/kernel/Regularizer/Squareб
"conv2d_48/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_48/kernel/Regularizer/Const┬
 conv2d_48/kernel/Regularizer/SumSum'conv2d_48/kernel/Regularizer/Square:y:0+conv2d_48/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/SumН
"conv2d_48/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_48/kernel/Regularizer/mul/x─
 conv2d_48/kernel/Regularizer/mulMul+conv2d_48/kernel/Regularizer/mul/x:output:0)conv2d_48/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_48/kernel/Regularizer/mul╤
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╨
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul√
IdentityIdentitydropout_50/dropout/Mul_1:z:0&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp3^conv2d_48/kernel/Regularizer/Square/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2h
2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2conv2d_48/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:` \
/
_output_shapes
:         KK
)
_user_specified_namelambda_16_input
∙
┬
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148844

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
и
╙
8__inference_batch_normalization_16_layer_call_fn_2148361

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallи
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_21441832
StatefulPartitionedCallЦ
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
└
о
E__inference_dense_40_layer_call_and_return_conditional_losses_2148575

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_40/kernel/Regularizer/Square/ReadVariableOpР
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╚
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp╣
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2$
"dense_40/kernel/Regularizer/SquareЧ
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const╛
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/SumЛ
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_40/kernel/Regularizer/mul/x└
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
─
b
F__inference_lambda_17_layer_call_and_return_conditional_losses_2148720

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
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
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
р
N
2__inference_max_pooling2d_48_layer_call_fn_2143772

inputs
identityє
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
GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_21437662
PartitionedCallП
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
щ
┤
__inference_loss_fn_2_2148694N
:dense_41_kernel_regularizer_square_readvariableop_resource:
АА
identityИв1dense_41/kernel/Regularizer/Square/ReadVariableOpу
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_41_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp╕
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_41/kernel/Regularizer/SquareЧ
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const╛
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/SumЛ
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_41/kernel/Regularizer/mul/x└
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulЪ
IdentityIdentity#dense_41/kernel/Regularizer/mul:z:02^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp
╨
в
+__inference_conv2d_49_layer_call_fn_2148474

inputs"
unknown: А
	unknown_0:	А
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_49_layer_call_and_return_conditional_losses_21438752
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         %%А2

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
ў
e
,__inference_dropout_51_layer_call_fn_2148926

inputs
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_51_layer_call_and_return_conditional_losses_21449872
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╦
H
,__inference_dropout_53_layer_call_fn_2149050

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_53_layer_call_and_return_conditional_losses_21448432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ш
e
G__inference_dropout_51_layer_call_and_return_conditional_losses_2148931

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         		А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
°
e
G__inference_dropout_52_layer_call_and_return_conditional_losses_2149001

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
C
input_18
serving_default_input_1:0         KK<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:╜┼
Щ

h2ptjl

h2ptj2
_output
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
ж__call__
+з&call_and_return_all_conditional_losses
и_default_save_signature
	йcall"Ы	
_tf_keras_modelБ	{"name": "CNN_2jet", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN_2jet", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 4]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN_2jet"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
фh

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
к__call__
+л&call_and_return_all_conditional_losses"¤d
_tf_keras_sequential▐d{"name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_16_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_16", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAWQChQJm\nBBkAUwApA07pAAAAAOkCAAAAqQCpAdoBeHIDAAAAcgMAAAD6Hy9ob21lL3NhbWh1YW5nL01ML0NO\nTi9tb2RlbHMucHnaCDxsYW1iZGE+KAEAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_49", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 4]}, "float32", "lambda_16_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_16_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_16", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAWQChQJm\nBBkAUwApA07pAAAAAOkCAAAAqQCpAdoBeHIDAAAAcgMAAAD6Hy9ob21lL3NhbWh1YW5nL01ML0NO\nTi9tb2RlbHMucHnaCDxsYW1iZGE+KAEAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_49", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
фh
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
 layer_with_weights-2
 layer-4
!layer-5
"layer_with_weights-3
"layer-6
#layer-7
$layer-8
%layer-9
&layer_with_weights-4
&layer-10
'layer-11
(layer_with_weights-5
(layer-12
)layer-13
*	variables
+trainable_variables
,regularization_losses
-	keras_api
м__call__
+н&call_and_return_all_conditional_losses"¤d
_tf_keras_sequential▐d{"name": "sequential_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_17_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_17", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAWQAhQJm\nBBkAUwCpAk7pAgAAAKkAqQHaAXhyAwAAAHIDAAAA+h8vaG9tZS9zYW1odWFuZy9NTC9DTk4vbW9k\nZWxzLnB52gg8bGFtYmRhPjoBAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_53", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_53", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 67, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 4]}, "float32", "lambda_17_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_17_input"}, "shared_object_id": 35}, {"class_name": "Lambda", "config": {"name": "lambda_17", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAWQAhQJm\nBBkAUwCpAk7pAgAAAKkAqQHaAXhyAwAAAHIDAAAA+h8vaG9tZS9zYW1odWFuZy9NTC9DTk4vbW9k\nZWxzLnB52gg8bGFtYmRhPjoBAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 36}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 38}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 40}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 41}, {"class_name": "Conv2D", "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 43}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 44}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 45}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 46}, {"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 49}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 50}, {"class_name": "Conv2D", "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 53}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_53", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 54}, {"class_name": "Dropout", "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 55}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 56}, {"class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 59}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60}, {"class_name": "Dropout", "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 61}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 62}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 63}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 64}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 65}, {"class_name": "Dropout", "config": {"name": "dropout_53", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 66}]}}}
┘

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
о__call__
+п&call_and_return_all_conditional_losses"▓
_tf_keras_layerШ{"name": "dense_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 68}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 69}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 70, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 1024]}}
█
4iter

5beta_1

6beta_2
	7decay
8learning_rate.mЄ/mє9mЇ:mї=mЎ>mў?m°@m∙Am·Bm√Cm№Dm¤Em■Fm GmАHmБKmВLmГMmДNmЕOmЖPmЗQmИRmЙSmКTmЛ.vМ/vН9vО:vП=vР>vС?vТ@vУAvФBvХCvЦDvЧEvШFvЩGvЪHvЫKvЬLvЭMvЮNvЯOvаPvбQvвRvгSvдTvе"
	optimizer
Ж
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
.28
/29"
trackable_list_wrapper
ц
90
:1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
.24
/25"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
	variables
trainable_variables
Umetrics
Vlayer_regularization_losses
Wlayer_metrics

Xlayers
regularization_losses
Ynon_trainable_variables
ж__call__
и_default_save_signature
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
-
░serving_default"
signature_map
ц
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"╒
_tf_keras_layer╗{"name": "lambda_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_16", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAWQChQJm\nBBkAUwApA07pAAAAAOkCAAAAqQCpAdoBeHIDAAAAcgMAAAD6Hy9ob21lL3NhbWh1YW5nL01ML0NO\nTi9tb2RlbHMucHnaCDxsYW1iZGE+KAEAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
╞

^axis
	9gamma
:beta
;moving_mean
<moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
в

=kernel
>bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses"√	
_tf_keras_layerс	{"name": "conv2d_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
│
g	variables
htrainable_variables
iregularization_losses
j	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"name": "max_pooling2d_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 74}}
╓


?kernel
@bias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"п	
_tf_keras_layerХ	{"name": "conv2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
│
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"name": "max_pooling2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 76}}
╪


Akernel
Bbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses"▒	
_tf_keras_layerЧ	{"name": "conv2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
│
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"name": "max_pooling2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 78}}
Б
{	variables
|trainable_variables
}regularization_losses
~	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "dropout_48", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_48", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
Э
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
├__call__
+─&call_and_return_all_conditional_losses"Й
_tf_keras_layerя{"name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 79}}
о	

Ckernel
Dbias
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses"Г
_tf_keras_layerщ{"name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
Е
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "dropout_49", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_49", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
к	

Ekernel
Fbias
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses" 
_tf_keras_layerх{"name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Е
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "dropout_50", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
Ж
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13"
trackable_list_wrapper
v
90
:1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11"
trackable_list_wrapper
8
═0
╬1
╧2"
trackable_list_wrapper
╡
	variables
trainable_variables
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
Цlayers
regularization_losses
Чnon_trainable_variables
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
ч
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
╨__call__
+╤&call_and_return_all_conditional_losses"╥
_tf_keras_layer╕{"name": "lambda_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_17", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAWQAhQJm\nBBkAUwCpAk7pAgAAAKkAqQHaAXhyAwAAAHIDAAAA+h8vaG9tZS9zYW1odWFuZy9NTC9DTk4vbW9k\nZWxzLnB52gg8bGFtYmRhPjoBAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 36}
╨

	Ьaxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
╥__call__
+╙&call_and_return_all_conditional_losses"ї
_tf_keras_layer█{"name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 38}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 40}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 82}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
з

Kkernel
Lbias
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
╘__call__
+╒&call_and_return_all_conditional_losses"№	
_tf_keras_layerт	{"name": "conv2d_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 43}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 44}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 83}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
╖
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
╓__call__
+╫&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"name": "max_pooling2d_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 84}}
┌


Mkernel
Nbias
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
╪__call__
+┘&call_and_return_all_conditional_losses"п	
_tf_keras_layerХ	{"name": "conv2d_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 85}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
╖
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
┌__call__
+█&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"name": "max_pooling2d_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 86}}
▄


Okernel
Pbias
▒	variables
▓trainable_variables
│regularization_losses
┤	keras_api
▄__call__
+▌&call_and_return_all_conditional_losses"▒	
_tf_keras_layerЧ	{"name": "conv2d_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 53, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 87}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
╖
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
▐__call__
+▀&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"name": "max_pooling2d_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_53", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 88}}
Е
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
р__call__
+с&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "dropout_51", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 55}
Ю
╜	variables
╛trainable_variables
┐regularization_losses
└	keras_api
т__call__
+у&call_and_return_all_conditional_losses"Й
_tf_keras_layerя{"name": "flatten_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 56, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 89}}
о	

Qkernel
Rbias
┴	variables
┬trainable_variables
├regularization_losses
─	keras_api
ф__call__
+х&call_and_return_all_conditional_losses"Г
_tf_keras_layerщ{"name": "dense_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 59}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 90}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
Е
┼	variables
╞trainable_variables
╟regularization_losses
╚	keras_api
ц__call__
+ч&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "dropout_52", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 61}
к	

Skernel
Tbias
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses" 
_tf_keras_layerх{"name": "dense_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 62}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 63}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 64}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 65, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 91}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Е
═	variables
╬trainable_variables
╧regularization_losses
╨	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "dropout_53", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_53", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 66}
Ж
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13"
trackable_list_wrapper
v
G0
H1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11"
trackable_list_wrapper
8
ь0
э1
ю2"
trackable_list_wrapper
╡
*	variables
+trainable_variables
╤metrics
 ╥layer_regularization_losses
╙layer_metrics
╘layers
,regularization_losses
╒non_trainable_variables
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_44/kernel
:2dense_44/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
0	variables
1trainable_variables
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
┘layers
2regularization_losses
┌non_trainable_variables
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
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
*:( 2conv2d_48/kernel
: 2conv2d_48/bias
+:) А2conv2d_49/kernel
:А2conv2d_49/bias
,:*АА2conv2d_50/kernel
:А2conv2d_50/bias
$:"АвА2dense_40/kernel
:А2dense_40/bias
#:!
АА2dense_41/kernel
:А2dense_41/bias
*:(2batch_normalization_17/gamma
):'2batch_normalization_17/beta
2:0 (2"batch_normalization_17/moving_mean
6:4 (2&batch_normalization_17/moving_variance
*:( 2conv2d_51/kernel
: 2conv2d_51/bias
+:) А2conv2d_52/kernel
:А2conv2d_52/bias
,:*АА2conv2d_53/kernel
:А2conv2d_53/bias
$:"АвА2dense_42/kernel
:А2dense_42/bias
#:!
АА2dense_43/kernel
:А2dense_43/bias
0
█0
▄1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
<
;0
<1
I2
J3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Z	variables
[trainable_variables
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
рlayers
\regularization_losses
сnon_trainable_variables
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
╡
_	variables
`trainable_variables
тmetrics
 уlayer_regularization_losses
фlayer_metrics
хlayers
aregularization_losses
цnon_trainable_variables
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
(
═0"
trackable_list_wrapper
╡
c	variables
dtrainable_variables
чmetrics
 шlayer_regularization_losses
щlayer_metrics
ъlayers
eregularization_losses
ыnon_trainable_variables
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
g	variables
htrainable_variables
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
яlayers
iregularization_losses
Ёnon_trainable_variables
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
k	variables
ltrainable_variables
ёmetrics
 Єlayer_regularization_losses
єlayer_metrics
Їlayers
mregularization_losses
їnon_trainable_variables
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
o	variables
ptrainable_variables
Ўmetrics
 ўlayer_regularization_losses
°layer_metrics
∙layers
qregularization_losses
·non_trainable_variables
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
s	variables
ttrainable_variables
√metrics
 №layer_regularization_losses
¤layer_metrics
■layers
uregularization_losses
 non_trainable_variables
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
w	variables
xtrainable_variables
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
Гlayers
yregularization_losses
Дnon_trainable_variables
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
{	variables
|trainable_variables
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
Иlayers
}regularization_losses
Йnon_trainable_variables
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╖
	variables
Аtrainable_variables
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
Нlayers
Бregularization_losses
Оnon_trainable_variables
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
(
╬0"
trackable_list_wrapper
╕
Г	variables
Дtrainable_variables
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Тlayers
Еregularization_losses
Уnon_trainable_variables
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
З	variables
Иtrainable_variables
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
Чlayers
Йregularization_losses
Шnon_trainable_variables
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
(
╧0"
trackable_list_wrapper
╕
Л	variables
Мtrainable_variables
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
Ьlayers
Нregularization_losses
Эnon_trainable_variables
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
П	variables
Рtrainable_variables
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
бlayers
Сregularization_losses
вnon_trainable_variables
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ж

0
1
2
3
4
5
6
7
8
9
10
11
12
13"
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
trackable_list_wrapper
╕
Ш	variables
Щtrainable_variables
гmetrics
 дlayer_regularization_losses
еlayer_metrics
жlayers
Ъregularization_losses
зnon_trainable_variables
╨__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
G0
H1
I2
J3"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Э	variables
Юtrainable_variables
иmetrics
 йlayer_regularization_losses
кlayer_metrics
лlayers
Яregularization_losses
мnon_trainable_variables
╥__call__
+╙&call_and_return_all_conditional_losses
'╙"call_and_return_conditional_losses"
_generic_user_object
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
(
ь0"
trackable_list_wrapper
╕
б	variables
вtrainable_variables
нmetrics
 оlayer_regularization_losses
пlayer_metrics
░layers
гregularization_losses
▒non_trainable_variables
╘__call__
+╒&call_and_return_all_conditional_losses
'╒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
е	variables
жtrainable_variables
▓metrics
 │layer_regularization_losses
┤layer_metrics
╡layers
зregularization_losses
╢non_trainable_variables
╓__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
й	variables
кtrainable_variables
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
║layers
лregularization_losses
╗non_trainable_variables
╪__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
н	variables
оtrainable_variables
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
┐layers
пregularization_losses
└non_trainable_variables
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▒	variables
▓trainable_variables
┴metrics
 ┬layer_regularization_losses
├layer_metrics
─layers
│regularization_losses
┼non_trainable_variables
▄__call__
+▌&call_and_return_all_conditional_losses
'▌"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╡	variables
╢trainable_variables
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
╔layers
╖regularization_losses
╩non_trainable_variables
▐__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╣	variables
║trainable_variables
╦metrics
 ╠layer_regularization_losses
═layer_metrics
╬layers
╗regularization_losses
╧non_trainable_variables
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╜	variables
╛trainable_variables
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
╙layers
┐regularization_losses
╘non_trainable_variables
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
(
э0"
trackable_list_wrapper
╕
┴	variables
┬trainable_variables
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
╪layers
├regularization_losses
┘non_trainable_variables
ф__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┼	variables
╞trainable_variables
┌metrics
 █layer_regularization_losses
▄layer_metrics
▌layers
╟regularization_losses
▐non_trainable_variables
ц__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
(
ю0"
trackable_list_wrapper
╕
╔	variables
╩trainable_variables
▀metrics
 рlayer_regularization_losses
сlayer_metrics
тlayers
╦regularization_losses
уnon_trainable_variables
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
═	variables
╬trainable_variables
фmetrics
 хlayer_regularization_losses
цlayer_metrics
чlayers
╧regularization_losses
шnon_trainable_variables
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ж
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12
)13"
trackable_list_wrapper
.
I0
J1"
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
╪

щtotal

ъcount
ы	variables
ь	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 92}
Ы

эtotal

юcount
я
_fn_kwargs
Ё	variables
ё	keras_api"╧
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
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
═0"
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
╬0"
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
╧0"
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
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
ь0"
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
э0"
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
ю0"
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
щ0
ъ1"
trackable_list_wrapper
.
ы	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
э0
ю1"
trackable_list_wrapper
.
Ё	variables"
_generic_user_object
':%	А2Adam/dense_44/kernel/m
 :2Adam/dense_44/bias/m
/:-2#Adam/batch_normalization_16/gamma/m
.:,2"Adam/batch_normalization_16/beta/m
/:- 2Adam/conv2d_48/kernel/m
!: 2Adam/conv2d_48/bias/m
0:. А2Adam/conv2d_49/kernel/m
": А2Adam/conv2d_49/bias/m
1:/АА2Adam/conv2d_50/kernel/m
": А2Adam/conv2d_50/bias/m
):'АвА2Adam/dense_40/kernel/m
!:А2Adam/dense_40/bias/m
(:&
АА2Adam/dense_41/kernel/m
!:А2Adam/dense_41/bias/m
/:-2#Adam/batch_normalization_17/gamma/m
.:,2"Adam/batch_normalization_17/beta/m
/:- 2Adam/conv2d_51/kernel/m
!: 2Adam/conv2d_51/bias/m
0:. А2Adam/conv2d_52/kernel/m
": А2Adam/conv2d_52/bias/m
1:/АА2Adam/conv2d_53/kernel/m
": А2Adam/conv2d_53/bias/m
):'АвА2Adam/dense_42/kernel/m
!:А2Adam/dense_42/bias/m
(:&
АА2Adam/dense_43/kernel/m
!:А2Adam/dense_43/bias/m
':%	А2Adam/dense_44/kernel/v
 :2Adam/dense_44/bias/v
/:-2#Adam/batch_normalization_16/gamma/v
.:,2"Adam/batch_normalization_16/beta/v
/:- 2Adam/conv2d_48/kernel/v
!: 2Adam/conv2d_48/bias/v
0:. А2Adam/conv2d_49/kernel/v
": А2Adam/conv2d_49/bias/v
1:/АА2Adam/conv2d_50/kernel/v
": А2Adam/conv2d_50/bias/v
):'АвА2Adam/dense_40/kernel/v
!:А2Adam/dense_40/bias/v
(:&
АА2Adam/dense_41/kernel/v
!:А2Adam/dense_41/bias/v
/:-2#Adam/batch_normalization_17/gamma/v
.:,2"Adam/batch_normalization_17/beta/v
/:- 2Adam/conv2d_51/kernel/v
!: 2Adam/conv2d_51/bias/v
0:. А2Adam/conv2d_52/kernel/v
": А2Adam/conv2d_52/bias/v
1:/АА2Adam/conv2d_53/kernel/v
": А2Adam/conv2d_53/bias/v
):'АвА2Adam/dense_42/kernel/v
!:А2Adam/dense_42/bias/v
(:&
АА2Adam/dense_43/kernel/v
!:А2Adam/dense_43/bias/v
ъ2ч
*__inference_CNN_2jet_layer_call_fn_2146252
*__inference_CNN_2jet_layer_call_fn_2146317
*__inference_CNN_2jet_layer_call_fn_2146382
*__inference_CNN_2jet_layer_call_fn_2146447┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2146618
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2146831
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2147002
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2147215┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ш2х
"__inference__wrapped_model_2143634╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_1         KK
З2Д
__inference_call_1988836
__inference_call_1988971
__inference_call_1989106│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
К2З
/__inference_sequential_16_layer_call_fn_2147266
/__inference_sequential_16_layer_call_fn_2147299
/__inference_sequential_16_layer_call_fn_2147332
/__inference_sequential_16_layer_call_fn_2147365└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ў2є
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147448
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147552
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147635
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147739└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
/__inference_sequential_17_layer_call_fn_2147790
/__inference_sequential_17_layer_call_fn_2147823
/__inference_sequential_17_layer_call_fn_2147856
/__inference_sequential_17_layer_call_fn_2147889└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ў2є
J__inference_sequential_17_layer_call_and_return_conditional_losses_2147972
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148076
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148159
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148263└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
*__inference_dense_44_layer_call_fn_2148272в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_44_layer_call_and_return_conditional_losses_2148283в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠B╔
%__inference_signature_wrapper_2146187input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
а2Э
+__inference_lambda_16_layer_call_fn_2148288
+__inference_lambda_16_layer_call_fn_2148293└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
F__inference_lambda_16_layer_call_and_return_conditional_losses_2148301
F__inference_lambda_16_layer_call_and_return_conditional_losses_2148309└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_16_layer_call_fn_2148322
8__inference_batch_normalization_16_layer_call_fn_2148335
8__inference_batch_normalization_16_layer_call_fn_2148348
8__inference_batch_normalization_16_layer_call_fn_2148361┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
О2Л
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148379
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148397
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148415
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148433┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╒2╥
+__inference_conv2d_48_layer_call_fn_2148448в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2148465в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
2__inference_max_pooling2d_48_layer_call_fn_2143772р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╡2▓
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2143766р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╒2╥
+__inference_conv2d_49_layer_call_fn_2148474в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_conv2d_49_layer_call_and_return_conditional_losses_2148485в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
2__inference_max_pooling2d_49_layer_call_fn_2143784р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╡2▓
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2143778р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╒2╥
+__inference_conv2d_50_layer_call_fn_2148494в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_conv2d_50_layer_call_and_return_conditional_losses_2148505в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
2__inference_max_pooling2d_50_layer_call_fn_2143796р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╡2▓
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2143790р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
,__inference_dropout_48_layer_call_fn_2148510
,__inference_dropout_48_layer_call_fn_2148515┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔
G__inference_dropout_48_layer_call_and_return_conditional_losses_2148520
G__inference_dropout_48_layer_call_and_return_conditional_losses_2148532┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
,__inference_flatten_16_layer_call_fn_2148537в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_flatten_16_layer_call_and_return_conditional_losses_2148543в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_40_layer_call_fn_2148558в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_40_layer_call_and_return_conditional_losses_2148575в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ц2У
,__inference_dropout_49_layer_call_fn_2148580
,__inference_dropout_49_layer_call_fn_2148585┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔
G__inference_dropout_49_layer_call_and_return_conditional_losses_2148590
G__inference_dropout_49_layer_call_and_return_conditional_losses_2148602┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
*__inference_dense_41_layer_call_fn_2148617в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_41_layer_call_and_return_conditional_losses_2148634в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ц2У
,__inference_dropout_50_layer_call_fn_2148639
,__inference_dropout_50_layer_call_fn_2148644┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔
G__inference_dropout_50_layer_call_and_return_conditional_losses_2148649
G__inference_dropout_50_layer_call_and_return_conditional_losses_2148661┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┤2▒
__inference_loss_fn_0_2148672П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
┤2▒
__inference_loss_fn_1_2148683П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
┤2▒
__inference_loss_fn_2_2148694П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
а2Э
+__inference_lambda_17_layer_call_fn_2148699
+__inference_lambda_17_layer_call_fn_2148704└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
F__inference_lambda_17_layer_call_and_return_conditional_losses_2148712
F__inference_lambda_17_layer_call_and_return_conditional_losses_2148720└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_17_layer_call_fn_2148733
8__inference_batch_normalization_17_layer_call_fn_2148746
8__inference_batch_normalization_17_layer_call_fn_2148759
8__inference_batch_normalization_17_layer_call_fn_2148772┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
О2Л
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148790
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148808
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148826
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148844┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╒2╥
+__inference_conv2d_51_layer_call_fn_2148859в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_conv2d_51_layer_call_and_return_conditional_losses_2148876в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
2__inference_max_pooling2d_51_layer_call_fn_2144642р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╡2▓
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2144636р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╒2╥
+__inference_conv2d_52_layer_call_fn_2148885в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_conv2d_52_layer_call_and_return_conditional_losses_2148896в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
2__inference_max_pooling2d_52_layer_call_fn_2144654р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╡2▓
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2144648р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╒2╥
+__inference_conv2d_53_layer_call_fn_2148905в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_conv2d_53_layer_call_and_return_conditional_losses_2148916в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
2__inference_max_pooling2d_53_layer_call_fn_2144666р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╡2▓
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_2144660р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
,__inference_dropout_51_layer_call_fn_2148921
,__inference_dropout_51_layer_call_fn_2148926┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔
G__inference_dropout_51_layer_call_and_return_conditional_losses_2148931
G__inference_dropout_51_layer_call_and_return_conditional_losses_2148943┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
,__inference_flatten_17_layer_call_fn_2148948в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_flatten_17_layer_call_and_return_conditional_losses_2148954в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_42_layer_call_fn_2148969в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_42_layer_call_and_return_conditional_losses_2148986в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ц2У
,__inference_dropout_52_layer_call_fn_2148991
,__inference_dropout_52_layer_call_fn_2148996┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔
G__inference_dropout_52_layer_call_and_return_conditional_losses_2149001
G__inference_dropout_52_layer_call_and_return_conditional_losses_2149013┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
*__inference_dense_43_layer_call_fn_2149028в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_43_layer_call_and_return_conditional_losses_2149045в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ц2У
,__inference_dropout_53_layer_call_fn_2149050
,__inference_dropout_53_layer_call_fn_2149055┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔
G__inference_dropout_53_layer_call_and_return_conditional_losses_2149060
G__inference_dropout_53_layer_call_and_return_conditional_losses_2149072┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┤2▒
__inference_loss_fn_3_2149083П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
┤2▒
__inference_loss_fn_4_2149094П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
┤2▒
__inference_loss_fn_5_2149105П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в ╬
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2146618Д9:;<=>?@ABCDEFGHIJKLMNOPQRST./;в8
1в.
(К%
inputs         KK
p 
к "%в"
К
0         
Ъ ╬
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2146831Д9:;<=>?@ABCDEFGHIJKLMNOPQRST./;в8
1в.
(К%
inputs         KK
p
к "%в"
К
0         
Ъ ╧
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2147002Е9:;<=>?@ABCDEFGHIJKLMNOPQRST./<в9
2в/
)К&
input_1         KK
p 
к "%в"
К
0         
Ъ ╧
E__inference_CNN_2jet_layer_call_and_return_conditional_losses_2147215Е9:;<=>?@ABCDEFGHIJKLMNOPQRST./<в9
2в/
)К&
input_1         KK
p
к "%в"
К
0         
Ъ ж
*__inference_CNN_2jet_layer_call_fn_2146252x9:;<=>?@ABCDEFGHIJKLMNOPQRST./<в9
2в/
)К&
input_1         KK
p 
к "К         е
*__inference_CNN_2jet_layer_call_fn_2146317w9:;<=>?@ABCDEFGHIJKLMNOPQRST./;в8
1в.
(К%
inputs         KK
p 
к "К         е
*__inference_CNN_2jet_layer_call_fn_2146382w9:;<=>?@ABCDEFGHIJKLMNOPQRST./;в8
1в.
(К%
inputs         KK
p
к "К         ж
*__inference_CNN_2jet_layer_call_fn_2146447x9:;<=>?@ABCDEFGHIJKLMNOPQRST./<в9
2в/
)К&
input_1         KK
p
к "К         ╢
"__inference__wrapped_model_2143634П9:;<=>?@ABCDEFGHIJKLMNOPQRST./8в5
.в+
)К&
input_1         KK
к "3к0
.
output_1"К
output_1         ю
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148379Ц9:;<MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ю
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148397Ц9:;<MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╔
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148415r9:;<;в8
1в.
(К%
inputs         KK
p 
к "-в*
#К 
0         KK
Ъ ╔
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2148433r9:;<;в8
1в.
(К%
inputs         KK
p
к "-в*
#К 
0         KK
Ъ ╞
8__inference_batch_normalization_16_layer_call_fn_2148322Й9:;<MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ╞
8__inference_batch_normalization_16_layer_call_fn_2148335Й9:;<MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           б
8__inference_batch_normalization_16_layer_call_fn_2148348e9:;<;в8
1в.
(К%
inputs         KK
p 
к " К         KKб
8__inference_batch_normalization_16_layer_call_fn_2148361e9:;<;в8
1в.
(К%
inputs         KK
p
к " К         KKю
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148790ЦGHIJMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ю
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148808ЦGHIJMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╔
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148826rGHIJ;в8
1в.
(К%
inputs         KK
p 
к "-в*
#К 
0         KK
Ъ ╔
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2148844rGHIJ;в8
1в.
(К%
inputs         KK
p
к "-в*
#К 
0         KK
Ъ ╞
8__inference_batch_normalization_17_layer_call_fn_2148733ЙGHIJMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ╞
8__inference_batch_normalization_17_layer_call_fn_2148746ЙGHIJMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           б
8__inference_batch_normalization_17_layer_call_fn_2148759eGHIJ;в8
1в.
(К%
inputs         KK
p 
к " К         KKб
8__inference_batch_normalization_17_layer_call_fn_2148772eGHIJ;в8
1в.
(К%
inputs         KK
p
к " К         KKГ
__inference_call_1988836g9:;<=>?@ABCDEFGHIJKLMNOPQRST./3в0
)в&
 К
inputsАKK
p
к "К	АГ
__inference_call_1988971g9:;<=>?@ABCDEFGHIJKLMNOPQRST./3в0
)в&
 К
inputsАKK
p 
к "К	АУ
__inference_call_1989106w9:;<=>?@ABCDEFGHIJKLMNOPQRST./;в8
1в.
(К%
inputs         KK
p 
к "К         ╢
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2148465l=>7в4
-в*
(К%
inputs         KK
к "-в*
#К 
0         KK 
Ъ О
+__inference_conv2d_48_layer_call_fn_2148448_=>7в4
-в*
(К%
inputs         KK
к " К         KK ╖
F__inference_conv2d_49_layer_call_and_return_conditional_losses_2148485m?@7в4
-в*
(К%
inputs         %% 
к ".в+
$К!
0         %%А
Ъ П
+__inference_conv2d_49_layer_call_fn_2148474`?@7в4
-в*
(К%
inputs         %% 
к "!К         %%А╕
F__inference_conv2d_50_layer_call_and_return_conditional_losses_2148505nAB8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Р
+__inference_conv2d_50_layer_call_fn_2148494aAB8в5
.в+
)К&
inputs         А
к "!К         А╢
F__inference_conv2d_51_layer_call_and_return_conditional_losses_2148876lKL7в4
-в*
(К%
inputs         KK
к "-в*
#К 
0         KK 
Ъ О
+__inference_conv2d_51_layer_call_fn_2148859_KL7в4
-в*
(К%
inputs         KK
к " К         KK ╖
F__inference_conv2d_52_layer_call_and_return_conditional_losses_2148896mMN7в4
-в*
(К%
inputs         %% 
к ".в+
$К!
0         %%А
Ъ П
+__inference_conv2d_52_layer_call_fn_2148885`MN7в4
-в*
(К%
inputs         %% 
к "!К         %%А╕
F__inference_conv2d_53_layer_call_and_return_conditional_losses_2148916nOP8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Р
+__inference_conv2d_53_layer_call_fn_2148905aOP8в5
.в+
)К&
inputs         А
к "!К         Аи
E__inference_dense_40_layer_call_and_return_conditional_losses_2148575_CD1в.
'в$
"К
inputs         Ав
к "&в#
К
0         А
Ъ А
*__inference_dense_40_layer_call_fn_2148558RCD1в.
'в$
"К
inputs         Ав
к "К         Аз
E__inference_dense_41_layer_call_and_return_conditional_losses_2148634^EF0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ 
*__inference_dense_41_layer_call_fn_2148617QEF0в-
&в#
!К
inputs         А
к "К         Аи
E__inference_dense_42_layer_call_and_return_conditional_losses_2148986_QR1в.
'в$
"К
inputs         Ав
к "&в#
К
0         А
Ъ А
*__inference_dense_42_layer_call_fn_2148969RQR1в.
'в$
"К
inputs         Ав
к "К         Аз
E__inference_dense_43_layer_call_and_return_conditional_losses_2149045^ST0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ 
*__inference_dense_43_layer_call_fn_2149028QST0в-
&в#
!К
inputs         А
к "К         Аж
E__inference_dense_44_layer_call_and_return_conditional_losses_2148283]./0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ ~
*__inference_dense_44_layer_call_fn_2148272P./0в-
&в#
!К
inputs         А
к "К         ╣
G__inference_dropout_48_layer_call_and_return_conditional_losses_2148520n<в9
2в/
)К&
inputs         		А
p 
к ".в+
$К!
0         		А
Ъ ╣
G__inference_dropout_48_layer_call_and_return_conditional_losses_2148532n<в9
2в/
)К&
inputs         		А
p
к ".в+
$К!
0         		А
Ъ С
,__inference_dropout_48_layer_call_fn_2148510a<в9
2в/
)К&
inputs         		А
p 
к "!К         		АС
,__inference_dropout_48_layer_call_fn_2148515a<в9
2в/
)К&
inputs         		А
p
к "!К         		Ай
G__inference_dropout_49_layer_call_and_return_conditional_losses_2148590^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ й
G__inference_dropout_49_layer_call_and_return_conditional_losses_2148602^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Б
,__inference_dropout_49_layer_call_fn_2148580Q4в1
*в'
!К
inputs         А
p 
к "К         АБ
,__inference_dropout_49_layer_call_fn_2148585Q4в1
*в'
!К
inputs         А
p
к "К         Ай
G__inference_dropout_50_layer_call_and_return_conditional_losses_2148649^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ й
G__inference_dropout_50_layer_call_and_return_conditional_losses_2148661^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Б
,__inference_dropout_50_layer_call_fn_2148639Q4в1
*в'
!К
inputs         А
p 
к "К         АБ
,__inference_dropout_50_layer_call_fn_2148644Q4в1
*в'
!К
inputs         А
p
к "К         А╣
G__inference_dropout_51_layer_call_and_return_conditional_losses_2148931n<в9
2в/
)К&
inputs         		А
p 
к ".в+
$К!
0         		А
Ъ ╣
G__inference_dropout_51_layer_call_and_return_conditional_losses_2148943n<в9
2в/
)К&
inputs         		А
p
к ".в+
$К!
0         		А
Ъ С
,__inference_dropout_51_layer_call_fn_2148921a<в9
2в/
)К&
inputs         		А
p 
к "!К         		АС
,__inference_dropout_51_layer_call_fn_2148926a<в9
2в/
)К&
inputs         		А
p
к "!К         		Ай
G__inference_dropout_52_layer_call_and_return_conditional_losses_2149001^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ й
G__inference_dropout_52_layer_call_and_return_conditional_losses_2149013^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Б
,__inference_dropout_52_layer_call_fn_2148991Q4в1
*в'
!К
inputs         А
p 
к "К         АБ
,__inference_dropout_52_layer_call_fn_2148996Q4в1
*в'
!К
inputs         А
p
к "К         Ай
G__inference_dropout_53_layer_call_and_return_conditional_losses_2149060^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ й
G__inference_dropout_53_layer_call_and_return_conditional_losses_2149072^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Б
,__inference_dropout_53_layer_call_fn_2149050Q4в1
*в'
!К
inputs         А
p 
к "К         АБ
,__inference_dropout_53_layer_call_fn_2149055Q4в1
*в'
!К
inputs         А
p
к "К         Ао
G__inference_flatten_16_layer_call_and_return_conditional_losses_2148543c8в5
.в+
)К&
inputs         		А
к "'в$
К
0         Ав
Ъ Ж
,__inference_flatten_16_layer_call_fn_2148537V8в5
.в+
)К&
inputs         		А
к "К         Аво
G__inference_flatten_17_layer_call_and_return_conditional_losses_2148954c8в5
.в+
)К&
inputs         		А
к "'в$
К
0         Ав
Ъ Ж
,__inference_flatten_17_layer_call_fn_2148948V8в5
.в+
)К&
inputs         		А
к "К         Ав║
F__inference_lambda_16_layer_call_and_return_conditional_losses_2148301p?в<
5в2
(К%
inputs         KK

 
p 
к "-в*
#К 
0         KK
Ъ ║
F__inference_lambda_16_layer_call_and_return_conditional_losses_2148309p?в<
5в2
(К%
inputs         KK

 
p
к "-в*
#К 
0         KK
Ъ Т
+__inference_lambda_16_layer_call_fn_2148288c?в<
5в2
(К%
inputs         KK

 
p 
к " К         KKТ
+__inference_lambda_16_layer_call_fn_2148293c?в<
5в2
(К%
inputs         KK

 
p
к " К         KK║
F__inference_lambda_17_layer_call_and_return_conditional_losses_2148712p?в<
5в2
(К%
inputs         KK

 
p 
к "-в*
#К 
0         KK
Ъ ║
F__inference_lambda_17_layer_call_and_return_conditional_losses_2148720p?в<
5в2
(К%
inputs         KK

 
p
к "-в*
#К 
0         KK
Ъ Т
+__inference_lambda_17_layer_call_fn_2148699c?в<
5в2
(К%
inputs         KK

 
p 
к " К         KKТ
+__inference_lambda_17_layer_call_fn_2148704c?в<
5в2
(К%
inputs         KK

 
p
к " К         KK<
__inference_loss_fn_0_2148672=в

в 
к "К <
__inference_loss_fn_1_2148683Cв

в 
к "К <
__inference_loss_fn_2_2148694Eв

в 
к "К <
__inference_loss_fn_3_2149083Kв

в 
к "К <
__inference_loss_fn_4_2149094Qв

в 
к "К <
__inference_loss_fn_5_2149105Sв

в 
к "К Ё
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2143766ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_48_layer_call_fn_2143772СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2143778ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_49_layer_call_fn_2143784СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2143790ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_50_layer_call_fn_2143796СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2144636ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_51_layer_call_fn_2144642СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2144648ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_52_layer_call_fn_2144654СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_53_layer_call_and_return_conditional_losses_2144660ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_53_layer_call_fn_2144666СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╟
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147448y9:;<=>?@ABCDEF?в<
5в2
(К%
inputs         KK
p 

 
к "&в#
К
0         А
Ъ ╟
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147552y9:;<=>?@ABCDEF?в<
5в2
(К%
inputs         KK
p

 
к "&в#
К
0         А
Ъ ╤
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147635В9:;<=>?@ABCDEFHвE
>в;
1К.
lambda_16_input         KK
p 

 
к "&в#
К
0         А
Ъ ╤
J__inference_sequential_16_layer_call_and_return_conditional_losses_2147739В9:;<=>?@ABCDEFHвE
>в;
1К.
lambda_16_input         KK
p

 
к "&в#
К
0         А
Ъ и
/__inference_sequential_16_layer_call_fn_2147266u9:;<=>?@ABCDEFHвE
>в;
1К.
lambda_16_input         KK
p 

 
к "К         АЯ
/__inference_sequential_16_layer_call_fn_2147299l9:;<=>?@ABCDEF?в<
5в2
(К%
inputs         KK
p 

 
к "К         АЯ
/__inference_sequential_16_layer_call_fn_2147332l9:;<=>?@ABCDEF?в<
5в2
(К%
inputs         KK
p

 
к "К         Аи
/__inference_sequential_16_layer_call_fn_2147365u9:;<=>?@ABCDEFHвE
>в;
1К.
lambda_16_input         KK
p

 
к "К         А╟
J__inference_sequential_17_layer_call_and_return_conditional_losses_2147972yGHIJKLMNOPQRST?в<
5в2
(К%
inputs         KK
p 

 
к "&в#
К
0         А
Ъ ╟
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148076yGHIJKLMNOPQRST?в<
5в2
(К%
inputs         KK
p

 
к "&в#
К
0         А
Ъ ╤
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148159ВGHIJKLMNOPQRSTHвE
>в;
1К.
lambda_17_input         KK
p 

 
к "&в#
К
0         А
Ъ ╤
J__inference_sequential_17_layer_call_and_return_conditional_losses_2148263ВGHIJKLMNOPQRSTHвE
>в;
1К.
lambda_17_input         KK
p

 
к "&в#
К
0         А
Ъ и
/__inference_sequential_17_layer_call_fn_2147790uGHIJKLMNOPQRSTHвE
>в;
1К.
lambda_17_input         KK
p 

 
к "К         АЯ
/__inference_sequential_17_layer_call_fn_2147823lGHIJKLMNOPQRST?в<
5в2
(К%
inputs         KK
p 

 
к "К         АЯ
/__inference_sequential_17_layer_call_fn_2147856lGHIJKLMNOPQRST?в<
5в2
(К%
inputs         KK
p

 
к "К         Аи
/__inference_sequential_17_layer_call_fn_2147889uGHIJKLMNOPQRSTHвE
>в;
1К.
lambda_17_input         KK
p

 
к "К         А─
%__inference_signature_wrapper_2146187Ъ9:;<=>?@ABCDEFGHIJKLMNOPQRST./Cв@
в 
9к6
4
input_1)К&
input_1         KK"3к0
.
output_1"К
output_1         