─н!
ф┤
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
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718╧╤
{
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_29/kernel
t
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes
:	А*
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
О
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_9/gamma
З
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_9/beta
Е
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:*
dtype0
Д
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
Е
conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*!
shared_nameconv2d_37/kernel
~
$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*'
_output_shapes
: А*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_38/kernel

$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_38/bias
n
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_39/kernel

$conv2d_39/kernel/Read/ReadVariableOpReadVariableOpconv2d_39/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_39/bias
n
"conv2d_39/bias/Read/ReadVariableOpReadVariableOpconv2d_39/bias*
_output_shapes	
:А*
dtype0
|
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А* 
shared_namedense_27/kernel
u
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel* 
_output_shapes
:
А@А*
dtype0
s
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_27/bias
l
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes	
:А*
dtype0
|
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_28/kernel
u
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_28/bias
l
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes	
:А*
dtype0
Ъ
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_9/moving_mean
У
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_9/moving_variance
Ы
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
Й
Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_29/kernel/m
В
*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m*
_output_shapes
:	А*
dtype0
А
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
Ь
"Adam/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_9/gamma/m
Х
6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_9/beta/m
У
5Adam/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_36/kernel/m
Л
+Adam/conv2d_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/m*&
_output_shapes
: *
dtype0
В
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
У
Adam/conv2d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_37/kernel/m
М
+Adam/conv2d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/m*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_37/bias/m
|
)Adam/conv2d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_38/kernel/m
Н
+Adam/conv2d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_38/bias/m
|
)Adam/conv2d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_39/kernel/m
Н
+Adam/conv2d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_39/bias/m
|
)Adam/conv2d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_27/kernel/m
Г
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_27/bias/m
z
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_28/kernel/m
Г
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_28/bias/m
z
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_29/kernel/v
В
*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v*
_output_shapes
:	А*
dtype0
А
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
Ь
"Adam/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_9/gamma/v
Х
6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_9/beta/v
У
5Adam/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_36/kernel/v
Л
+Adam/conv2d_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/v*&
_output_shapes
: *
dtype0
В
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
У
Adam/conv2d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_37/kernel/v
М
+Adam/conv2d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/v*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_37/bias/v
|
)Adam/conv2d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_38/kernel/v
Н
+Adam/conv2d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_38/bias/v
|
)Adam/conv2d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_39/kernel/v
Н
+Adam/conv2d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_39/bias/v
|
)Adam/conv2d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_27/kernel/v
Г
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_27/bias/v
z
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_28/kernel/v
Г
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_28/bias/v
z
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
Нl
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╚k
value╛kB╗k B┤k
К

h2ptjl
_output
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
▐
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
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
А
#iter

$beta_1

%beta_2
	&decay
'learning_ratemуmф(mх)mц*mч+mш,mщ-mъ.mы/mь0mэ1mю2mя3mЁ4mё5mЄvєvЇ(vї)vЎ*vў+v°,v∙-v·.v√/v№0v¤1v■2v 3vА4vБ5vВ
v
(0
)1
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
412
513
14
15
 
Ж
(0
)1
62
73
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
414
515
16
17
н
8metrics
9layer_metrics
trainable_variables
:non_trainable_variables
regularization_losses
	variables

;layers
<layer_regularization_losses
 
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
Ч
Aaxis
	(gamma
)beta
6moving_mean
7moving_variance
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
h

*kernel
+bias
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
R
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
h

,kernel
-bias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
R
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
h

.kernel
/bias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
R
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
h

0kernel
1bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
R
btrainable_variables
cregularization_losses
d	variables
e	keras_api
R
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
R
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
h

2kernel
3bias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
R
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
h

4kernel
5bias
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
R
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
f
(0
)1
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
412
513
 
v
(0
)1
62
73
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
414
515
░
~metrics
layer_metrics
trainable_variables
Аnon_trainable_variables
regularization_losses
	variables
Бlayers
 Вlayer_regularization_losses
NL
VARIABLE_VALUEdense_29/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_29/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
▓
Гmetrics
Дlayer_metrics
trainable_variables
Еnon_trainable_variables
 regularization_losses
!	variables
Жlayers
 Зlayer_regularization_losses
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
VARIABLE_VALUEconv2d_36/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_36/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_37/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_37/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_38/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_38/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_39/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_39/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_27/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_27/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_28/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_28/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_9/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_9/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE

И0
Й1
 

60
71

0
1
 
 
 
 
▓
Кmetrics
Лlayer_metrics
=trainable_variables
Мnon_trainable_variables
>regularization_losses
?	variables
Нlayers
 Оlayer_regularization_losses
 

(0
)1
 

(0
)1
62
73
▓
Пmetrics
Рlayer_metrics
Btrainable_variables
Сnon_trainable_variables
Cregularization_losses
D	variables
Тlayers
 Уlayer_regularization_losses

*0
+1
 

*0
+1
▓
Фmetrics
Хlayer_metrics
Ftrainable_variables
Цnon_trainable_variables
Gregularization_losses
H	variables
Чlayers
 Шlayer_regularization_losses
 
 
 
▓
Щmetrics
Ъlayer_metrics
Jtrainable_variables
Ыnon_trainable_variables
Kregularization_losses
L	variables
Ьlayers
 Эlayer_regularization_losses

,0
-1
 

,0
-1
▓
Юmetrics
Яlayer_metrics
Ntrainable_variables
аnon_trainable_variables
Oregularization_losses
P	variables
бlayers
 вlayer_regularization_losses
 
 
 
▓
гmetrics
дlayer_metrics
Rtrainable_variables
еnon_trainable_variables
Sregularization_losses
T	variables
жlayers
 зlayer_regularization_losses

.0
/1
 

.0
/1
▓
иmetrics
йlayer_metrics
Vtrainable_variables
кnon_trainable_variables
Wregularization_losses
X	variables
лlayers
 мlayer_regularization_losses
 
 
 
▓
нmetrics
оlayer_metrics
Ztrainable_variables
пnon_trainable_variables
[regularization_losses
\	variables
░layers
 ▒layer_regularization_losses

00
11
 

00
11
▓
▓metrics
│layer_metrics
^trainable_variables
┤non_trainable_variables
_regularization_losses
`	variables
╡layers
 ╢layer_regularization_losses
 
 
 
▓
╖metrics
╕layer_metrics
btrainable_variables
╣non_trainable_variables
cregularization_losses
d	variables
║layers
 ╗layer_regularization_losses
 
 
 
▓
╝metrics
╜layer_metrics
ftrainable_variables
╛non_trainable_variables
gregularization_losses
h	variables
┐layers
 └layer_regularization_losses
 
 
 
▓
┴metrics
┬layer_metrics
jtrainable_variables
├non_trainable_variables
kregularization_losses
l	variables
─layers
 ┼layer_regularization_losses

20
31
 

20
31
▓
╞metrics
╟layer_metrics
ntrainable_variables
╚non_trainable_variables
oregularization_losses
p	variables
╔layers
 ╩layer_regularization_losses
 
 
 
▓
╦metrics
╠layer_metrics
rtrainable_variables
═non_trainable_variables
sregularization_losses
t	variables
╬layers
 ╧layer_regularization_losses

40
51
 

40
51
▓
╨metrics
╤layer_metrics
vtrainable_variables
╥non_trainable_variables
wregularization_losses
x	variables
╙layers
 ╘layer_regularization_losses
 
 
 
▓
╒metrics
╓layer_metrics
ztrainable_variables
╫non_trainable_variables
{regularization_losses
|	variables
╪layers
 ┘layer_regularization_losses
 
 

60
71
v
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
 
 
 
 
 
 
8

┌total

█count
▄	variables
▌	keras_api
I

▐total

▀count
р
_fn_kwargs
с	variables
т	keras_api
 
 
 
 
 
 
 

60
71
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
┌0
█1

▄	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

▐0
▀1

с	variables
qo
VARIABLE_VALUEAdam/dense_29/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_29/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_9/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_36/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_36/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_37/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_37/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_38/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_38/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_39/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_39/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_27/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_27/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_28/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_28/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_29/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_29/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_9/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_36/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_36/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_37/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_37/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_38/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_38/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_39/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_39/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_27/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_27/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_28/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_28/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
╟
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_1308898
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ъ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp+Adam/conv2d_36/kernel/m/Read/ReadVariableOp)Adam/conv2d_36/bias/m/Read/ReadVariableOp+Adam/conv2d_37/kernel/m/Read/ReadVariableOp)Adam/conv2d_37/bias/m/Read/ReadVariableOp+Adam/conv2d_38/kernel/m/Read/ReadVariableOp)Adam/conv2d_38/bias/m/Read/ReadVariableOp+Adam/conv2d_39/kernel/m/Read/ReadVariableOp)Adam/conv2d_39/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp+Adam/conv2d_36/kernel/v/Read/ReadVariableOp)Adam/conv2d_36/bias/v/Read/ReadVariableOp+Adam/conv2d_37/kernel/v/Read/ReadVariableOp)Adam/conv2d_37/bias/v/Read/ReadVariableOp+Adam/conv2d_38/kernel/v/Read/ReadVariableOp)Adam/conv2d_38/bias/v/Read/ReadVariableOp+Adam/conv2d_39/kernel/v/Read/ReadVariableOp)Adam/conv2d_39/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOpConst*H
TinA
?2=	*
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
 __inference__traced_save_1310719
╔
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_29/kerneldense_29/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_9/gammabatch_normalization_9/betaconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/bias!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancetotalcounttotal_1count_1Adam/dense_29/kernel/mAdam/dense_29/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/conv2d_36/kernel/mAdam/conv2d_36/bias/mAdam/conv2d_37/kernel/mAdam/conv2d_37/bias/mAdam/conv2d_38/kernel/mAdam/conv2d_38/bias/mAdam/conv2d_39/kernel/mAdam/conv2d_39/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/vAdam/dense_29/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/conv2d_36/kernel/vAdam/conv2d_36/bias/vAdam/conv2d_37/kernel/vAdam/conv2d_37/bias/vAdam/conv2d_38/kernel/vAdam/conv2d_38/bias/vAdam/conv2d_39/kernel/vAdam/conv2d_39/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/v*G
Tin@
>2<*
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
#__inference__traced_restore_1310906╛─
р
N
2__inference_max_pooling2d_37_layer_call_fn_1307609

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
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_13076032
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
ж
╥
7__inference_batch_normalization_9_layer_call_fn_1310238

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallз
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
GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_13080522
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
д3
Т
A__inference_CNN3_layer_call_and_return_conditional_losses_1308479

inputs"
sequential_9_1308410:"
sequential_9_1308412:"
sequential_9_1308414:"
sequential_9_1308416:.
sequential_9_1308418: "
sequential_9_1308420: /
sequential_9_1308422: А#
sequential_9_1308424:	А0
sequential_9_1308426:АА#
sequential_9_1308428:	А0
sequential_9_1308430:АА#
sequential_9_1308432:	А(
sequential_9_1308434:
А@А#
sequential_9_1308436:	А(
sequential_9_1308438:
АА#
sequential_9_1308440:	А#
dense_29_1308455:	А
dense_29_1308457:
identityИв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpв dense_29/StatefulPartitionedCallв$sequential_9/StatefulPartitionedCallБ
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_1308410sequential_9_1308412sequential_9_1308414sequential_9_1308416sequential_9_1308418sequential_9_1308420sequential_9_1308422sequential_9_1308424sequential_9_1308426sequential_9_1308428sequential_9_1308430sequential_9_1308432sequential_9_1308434sequential_9_1308436sequential_9_1308438sequential_9_1308440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_13078492&
$sequential_9/StatefulPartitionedCall├
 dense_29/StatefulPartitionedCallStatefulPartitionedCall-sequential_9/StatefulPartitionedCall:output:0dense_29_1308455dense_29_1308457*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_13084542"
 dense_29/StatefulPartitionedCall┼
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1308418*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╜
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1308434* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╜
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1308438* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulф
IdentityIdentity)dense_29/StatefulPartitionedCall:output:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp!^dense_29/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
└
┤
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1310261

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpХ
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
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
█╨
Ь
A__inference_CNN3_layer_call_and_return_conditional_losses_1309115

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: АE
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_38_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_39_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	АH
4sequential_9_dense_27_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_27_biasadd_readvariableop_resource:	АH
4sequential_9_dense_28_matmul_readvariableop_resource:
ААD
5sequential_9_dense_28_biasadd_readvariableop_resource:	А:
'dense_29_matmul_readvariableop_resource:	А6
(dense_29_biasadd_readvariableop_resource:
identityИв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpвdense_29/BiasAdd/ReadVariableOpвdense_29/MatMul/ReadVariableOpв1sequential_9/batch_normalization_9/AssignNewValueв3sequential_9/batch_normalization_9/AssignNewValue_1вBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpвDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в1sequential_9/batch_normalization_9/ReadVariableOpв3sequential_9/batch_normalization_9/ReadVariableOp_1в-sequential_9/conv2d_36/BiasAdd/ReadVariableOpв,sequential_9/conv2d_36/Conv2D/ReadVariableOpв-sequential_9/conv2d_37/BiasAdd/ReadVariableOpв,sequential_9/conv2d_37/Conv2D/ReadVariableOpв-sequential_9/conv2d_38/BiasAdd/ReadVariableOpв,sequential_9/conv2d_38/Conv2D/ReadVariableOpв-sequential_9/conv2d_39/BiasAdd/ReadVariableOpв,sequential_9/conv2d_39/Conv2D/ReadVariableOpв,sequential_9/dense_27/BiasAdd/ReadVariableOpв+sequential_9/dense_27/MatMul/ReadVariableOpв,sequential_9/dense_28/BiasAdd/ReadVariableOpв+sequential_9/dense_28/MatMul/ReadVariableOpп
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack│
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1│
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2ы
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice▌
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpу
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1Р
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1╨
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<25
3sequential_9/batch_normalization_9/FusedBatchNormV3ё
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_9/batch_normalization_9/AssignNewValue¤
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_9/batch_normalization_9/AssignNewValue_1┌
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D╤
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpф
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_9/conv2d_36/BiasAddе
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_9/conv2d_36/Reluё
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool█
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D╥
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpх
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_9/conv2d_37/BiasAddж
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_9/conv2d_37/ReluЄ
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool▄
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D╥
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpх
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_9/conv2d_38/BiasAddж
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_9/conv2d_38/ReluЄ
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool▄
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D╥
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpх
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_9/conv2d_39/BiasAddж
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_9/conv2d_39/ReluЄ
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPoolУ
%sequential_9/dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2'
%sequential_9/dropout_27/dropout/Constь
#sequential_9/dropout_27/dropout/MulMul.sequential_9/max_pooling2d_39/MaxPool:output:0.sequential_9/dropout_27/dropout/Const:output:0*
T0*0
_output_shapes
:         А2%
#sequential_9/dropout_27/dropout/Mulм
%sequential_9/dropout_27/dropout/ShapeShape.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_27/dropout/ShapeЕ
<sequential_9/dropout_27/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_27/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02>
<sequential_9/dropout_27/dropout/random_uniform/RandomUniformе
.sequential_9/dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_9/dropout_27/dropout/GreaterEqual/yз
,sequential_9/dropout_27/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_27/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_27/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2.
,sequential_9/dropout_27/dropout/GreaterEqual╨
$sequential_9/dropout_27/dropout/CastCast0sequential_9/dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2&
$sequential_9/dropout_27/dropout/Castу
%sequential_9/dropout_27/dropout/Mul_1Mul'sequential_9/dropout_27/dropout/Mul:z:0(sequential_9/dropout_27/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2'
%sequential_9/dropout_27/dropout/Mul_1Н
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_9/flatten_9/Const╨
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/dropout/Mul_1:z:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_9/flatten_9/Reshape╤
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp╫
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/MatMul╧
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp┌
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/BiasAddЫ
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/ReluУ
%sequential_9/dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_28/dropout/Const▐
#sequential_9/dropout_28/dropout/MulMul(sequential_9/dense_27/Relu:activations:0.sequential_9/dropout_28/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_9/dropout_28/dropout/Mulж
%sequential_9/dropout_28/dropout/ShapeShape(sequential_9/dense_27/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_28/dropout/Shape¤
<sequential_9/dropout_28/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_28/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_9/dropout_28/dropout/random_uniform/RandomUniformе
.sequential_9/dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_28/dropout/GreaterEqual/yЯ
,sequential_9/dropout_28/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_28/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_28/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_9/dropout_28/dropout/GreaterEqual╚
$sequential_9/dropout_28/dropout/CastCast0sequential_9/dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_9/dropout_28/dropout/Cast█
%sequential_9/dropout_28/dropout/Mul_1Mul'sequential_9/dropout_28/dropout/Mul:z:0(sequential_9/dropout_28/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_9/dropout_28/dropout/Mul_1╤
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp┘
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/dropout/Mul_1:z:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/MatMul╧
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp┌
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/BiasAddЫ
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/ReluУ
%sequential_9/dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_29/dropout/Const▐
#sequential_9/dropout_29/dropout/MulMul(sequential_9/dense_28/Relu:activations:0.sequential_9/dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_9/dropout_29/dropout/Mulж
%sequential_9/dropout_29/dropout/ShapeShape(sequential_9/dense_28/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_29/dropout/Shape¤
<sequential_9/dropout_29/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_9/dropout_29/dropout/random_uniform/RandomUniformе
.sequential_9/dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_29/dropout/GreaterEqual/yЯ
,sequential_9/dropout_29/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_29/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_9/dropout_29/dropout/GreaterEqual╚
$sequential_9/dropout_29/dropout/CastCast0sequential_9/dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_9/dropout_29/dropout/Cast█
%sequential_9/dropout_29/dropout/Mul_1Mul'sequential_9/dropout_29/dropout/Mul:z:0(sequential_9/dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_9/dropout_29/dropout/Mul_1й
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_29/MatMul/ReadVariableOp▒
dense_29/MatMulMatMul)sequential_9/dropout_29/dropout/Mul_1:z:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/MatMulз
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpе
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_29/Softmaxц
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul▌
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul▌
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulф	
IdentityIdentitydense_29/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_12И
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2М
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
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╕
└
.__inference_sequential_9_layer_call_fn_1309994

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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А
identityИвStatefulPartitionedCall║
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
 *(
_output_shapes
:         А*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_13078492
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▀√
╔%
#__inference__traced_restore_1310906
file_prefix3
 assignvariableop_dense_29_kernel:	А.
 assignvariableop_1_dense_29_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_9_gamma:;
-assignvariableop_8_batch_normalization_9_beta:=
#assignvariableop_9_conv2d_36_kernel: 0
"assignvariableop_10_conv2d_36_bias: ?
$assignvariableop_11_conv2d_37_kernel: А1
"assignvariableop_12_conv2d_37_bias:	А@
$assignvariableop_13_conv2d_38_kernel:АА1
"assignvariableop_14_conv2d_38_bias:	А@
$assignvariableop_15_conv2d_39_kernel:АА1
"assignvariableop_16_conv2d_39_bias:	А7
#assignvariableop_17_dense_27_kernel:
А@А0
!assignvariableop_18_dense_27_bias:	А7
#assignvariableop_19_dense_28_kernel:
АА0
!assignvariableop_20_dense_28_bias:	АC
5assignvariableop_21_batch_normalization_9_moving_mean:G
9assignvariableop_22_batch_normalization_9_moving_variance:#
assignvariableop_23_total: #
assignvariableop_24_count: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: =
*assignvariableop_27_adam_dense_29_kernel_m:	А6
(assignvariableop_28_adam_dense_29_bias_m:D
6assignvariableop_29_adam_batch_normalization_9_gamma_m:C
5assignvariableop_30_adam_batch_normalization_9_beta_m:E
+assignvariableop_31_adam_conv2d_36_kernel_m: 7
)assignvariableop_32_adam_conv2d_36_bias_m: F
+assignvariableop_33_adam_conv2d_37_kernel_m: А8
)assignvariableop_34_adam_conv2d_37_bias_m:	АG
+assignvariableop_35_adam_conv2d_38_kernel_m:АА8
)assignvariableop_36_adam_conv2d_38_bias_m:	АG
+assignvariableop_37_adam_conv2d_39_kernel_m:АА8
)assignvariableop_38_adam_conv2d_39_bias_m:	А>
*assignvariableop_39_adam_dense_27_kernel_m:
А@А7
(assignvariableop_40_adam_dense_27_bias_m:	А>
*assignvariableop_41_adam_dense_28_kernel_m:
АА7
(assignvariableop_42_adam_dense_28_bias_m:	А=
*assignvariableop_43_adam_dense_29_kernel_v:	А6
(assignvariableop_44_adam_dense_29_bias_v:D
6assignvariableop_45_adam_batch_normalization_9_gamma_v:C
5assignvariableop_46_adam_batch_normalization_9_beta_v:E
+assignvariableop_47_adam_conv2d_36_kernel_v: 7
)assignvariableop_48_adam_conv2d_36_bias_v: F
+assignvariableop_49_adam_conv2d_37_kernel_v: А8
)assignvariableop_50_adam_conv2d_37_bias_v:	АG
+assignvariableop_51_adam_conv2d_38_kernel_v:АА8
)assignvariableop_52_adam_conv2d_38_bias_v:	АG
+assignvariableop_53_adam_conv2d_39_kernel_v:АА8
)assignvariableop_54_adam_conv2d_39_bias_v:	А>
*assignvariableop_55_adam_dense_27_kernel_v:
А@А7
(assignvariableop_56_adam_dense_27_bias_v:	А>
*assignvariableop_57_adam_dense_28_kernel_v:
АА7
(assignvariableop_58_adam_dense_28_bias_v:	А
identity_60ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*№
valueЄBя<B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЙ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*Н
valueГBА<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices┌
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ж
_output_shapesє
Ё::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_29_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_29_biasIdentity_1:output:0"/device:CPU:0*
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

Identity_7│
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_9_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▓
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_9_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9и
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_36_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10к
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_36_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11м
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_37_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12к
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_37_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13м
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_38_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14к
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_38_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15м
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_39_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16к
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_39_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17л
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_27_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18й
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_27_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19л
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_28_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20й
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_28_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╜
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_9_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22┴
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_9_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23б
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24б
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25г
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26г
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27▓
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_29_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28░
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_29_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╛
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_batch_normalization_9_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╜
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_batch_normalization_9_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_36_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_36_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_37_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_37_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35│
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_38_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▒
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_38_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37│
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_39_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_39_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_27_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_27_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41▓
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_28_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42░
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_28_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▓
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_29_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44░
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_29_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45╛
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_9_gamma_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╜
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_9_beta_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_36_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_36_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49│
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_37_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50▒
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_37_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51│
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_38_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52▒
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_38_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53│
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_39_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54▒
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_39_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55▓
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_27_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56░
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_27_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57▓
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_28_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58░
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_28_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЁ

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59у

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_60"#
identity_60Identity_60:output:0*Л
_input_shapesz
x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°
┴
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310186

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
Ш
e
G__inference_dropout_27_layer_call_and_return_conditional_losses_1307760

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╢
f
G__inference_dropout_28_layer_call_and_return_conditional_losses_1307937

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
хг
│
A__inference_CNN3_layer_call_and_return_conditional_losses_1309213
input_1H
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: АE
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_38_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_39_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	АH
4sequential_9_dense_27_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_27_biasadd_readvariableop_resource:	АH
4sequential_9_dense_28_matmul_readvariableop_resource:
ААD
5sequential_9_dense_28_biasadd_readvariableop_resource:	А:
'dense_29_matmul_readvariableop_resource:	А6
(dense_29_biasadd_readvariableop_resource:
identityИв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpвdense_29/BiasAdd/ReadVariableOpвdense_29/MatMul/ReadVariableOpвBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpвDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в1sequential_9/batch_normalization_9/ReadVariableOpв3sequential_9/batch_normalization_9/ReadVariableOp_1в-sequential_9/conv2d_36/BiasAdd/ReadVariableOpв,sequential_9/conv2d_36/Conv2D/ReadVariableOpв-sequential_9/conv2d_37/BiasAdd/ReadVariableOpв,sequential_9/conv2d_37/Conv2D/ReadVariableOpв-sequential_9/conv2d_38/BiasAdd/ReadVariableOpв,sequential_9/conv2d_38/Conv2D/ReadVariableOpв-sequential_9/conv2d_39/BiasAdd/ReadVariableOpв,sequential_9/conv2d_39/Conv2D/ReadVariableOpв,sequential_9/dense_27/BiasAdd/ReadVariableOpв+sequential_9/dense_27/MatMul/ReadVariableOpв,sequential_9/dense_28/BiasAdd/ReadVariableOpв+sequential_9/dense_28/MatMul/ReadVariableOpп
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack│
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1│
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2ь
#sequential_9/lambda_9/strided_sliceStridedSliceinput_12sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice▌
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpу
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1Р
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3┌
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D╤
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpф
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_9/conv2d_36/BiasAddе
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_9/conv2d_36/Reluё
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool█
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D╥
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpх
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_9/conv2d_37/BiasAddж
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_9/conv2d_37/ReluЄ
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool▄
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D╥
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpх
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_9/conv2d_38/BiasAddж
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_9/conv2d_38/ReluЄ
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool▄
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D╥
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpх
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_9/conv2d_39/BiasAddж
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_9/conv2d_39/ReluЄ
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool╗
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_9/dropout_27/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_9/flatten_9/Const╨
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_9/flatten_9/Reshape╤
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp╫
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/MatMul╧
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp┌
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/BiasAddЫ
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/Reluн
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_9/dropout_28/Identity╤
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp┘
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/MatMul╧
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp┌
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/BiasAddЫ
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/Reluн
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_9/dropout_29/Identityй
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_29/MatMul/ReadVariableOp▒
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/MatMulз
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpе
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_29/Softmaxц
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul▌
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul▌
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul·
IdentityIdentitydense_29/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2И
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2М
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
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
н
i
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1307603

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
Цж
│
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309717

inputs;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_36_conv2d_readvariableop_resource: 7
)conv2d_36_biasadd_readvariableop_resource: C
(conv2d_37_conv2d_readvariableop_resource: А8
)conv2d_37_biasadd_readvariableop_resource:	АD
(conv2d_38_conv2d_readvariableop_resource:АА8
)conv2d_38_biasadd_readvariableop_resource:	АD
(conv2d_39_conv2d_readvariableop_resource:АА8
)conv2d_39_biasadd_readvariableop_resource:	А;
'dense_27_matmul_readvariableop_resource:
А@А7
(dense_27_biasadd_readvariableop_resource:	А;
'dense_28_matmul_readvariableop_resource:
АА7
(dense_28_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_9/AssignNewValueв&batch_normalization_9/AssignNewValue_1в5batch_normalization_9/FusedBatchNormV3/ReadVariableOpв7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_9/ReadVariableOpв&batch_normalization_9/ReadVariableOp_1в conv2d_36/BiasAdd/ReadVariableOpвconv2d_36/Conv2D/ReadVariableOpв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв conv2d_37/BiasAdd/ReadVariableOpвconv2d_37/Conv2D/ReadVariableOpв conv2d_38/BiasAdd/ReadVariableOpвconv2d_38/Conv2D/ReadVariableOpв conv2d_39/BiasAdd/ReadVariableOpвconv2d_39/Conv2D/ReadVariableOpвdense_27/BiasAdd/ReadVariableOpвdense_27/MatMul/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpвdense_28/BiasAdd/ReadVariableOpвdense_28/MatMul/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpХ
lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_9/strided_slice/stackЩ
lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_9/strided_slice/stack_1Щ
lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_9/strided_slice/stack_2к
lambda_9/strided_sliceStridedSliceinputs%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_9/strided_slice╢
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOp╝
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1щ
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ї
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_9/FusedBatchNormV3░
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue╝
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1│
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOpх
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_36/Conv2Dк
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp░
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_36/Relu╩
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool┤
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_37/Conv2D/ReadVariableOp▌
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_37/Conv2Dл
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp▒
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_37/Relu╦
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool╡
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_38/Conv2D/ReadVariableOp▌
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_38/Conv2Dл
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_38/Relu╦
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool╡
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_39/Conv2D/ReadVariableOp▌
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
conv2d_39/Conv2Dл
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▒
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
conv2d_39/Relu╦
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPooly
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_27/dropout/Const╕
dropout_27/dropout/MulMul!max_pooling2d_39/MaxPool:output:0!dropout_27/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout_27/dropout/MulЕ
dropout_27/dropout/ShapeShape!max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shape▐
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype021
/dropout_27/dropout/random_uniform/RandomUniformЛ
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_27/dropout/GreaterEqual/yє
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2!
dropout_27/dropout/GreaterEqualй
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_27/dropout/Castп
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_27/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_9/ConstЬ
flatten_9/ReshapeReshapedropout_27/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_9/Reshapeк
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_27/MatMul/ReadVariableOpг
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_27/MatMulи
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_27/BiasAdd/ReadVariableOpж
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_27/Reluy
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_28/dropout/Constк
dropout_28/dropout/MulMuldense_27/Relu:activations:0!dropout_28/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_28/dropout/Mul
dropout_28/dropout/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
dropout_28/dropout/Shape╓
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_28/dropout/random_uniform/RandomUniformЛ
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_28/dropout/GreaterEqual/yы
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_28/dropout/GreaterEqualб
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_28/dropout/Castз
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_28/dropout/Mul_1к
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_28/MatMul/ReadVariableOpе
dense_28/MatMulMatMuldropout_28/dropout/Mul_1:z:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_28/MatMulи
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_28/BiasAdd/ReadVariableOpж
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_28/Reluy
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_29/dropout/Constк
dropout_29/dropout/MulMuldense_28/Relu:activations:0!dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_29/dropout/Mul
dropout_29/dropout/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:2
dropout_29/dropout/Shape╓
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_29/dropout/random_uniform/RandomUniformЛ
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_29/dropout/GreaterEqual/yы
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_29/dropout/GreaterEqualб
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_29/dropout/Castз
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_29/dropout/Mul_1┘
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╨
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╨
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul║
IdentityIdentitydropout_29/dropout/Mul_1:z:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2L
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
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
и
╥
7__inference_batch_normalization_9_layer_call_fn_1310225

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallй
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
GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_13076672
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
щ
┤
__inference_loss_fn_1_1310508N
:dense_27_kernel_regularizer_square_readvariableop_resource:
А@А
identityИв1dense_27/kernel/Regularizer/Square/ReadVariableOpу
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_27_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mulЪ
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
е
Ш
*__inference_dense_29_layer_call_fn_1310088

inputs
unknown:	А
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
E__inference_dense_29_layer_call_and_return_conditional_losses_13084542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

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
─
Э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310168

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
¤
ё
&__inference_CNN3_layer_call_fn_1309455

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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:	А

unknown_16:
identityИвStatefulPartitionedCall╦
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
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_13086312
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

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
в
В
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1310321

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
Ш
e
G__inference_dropout_27_layer_call_and_return_conditional_losses_1310335

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╢
f
G__inference_dropout_29_layer_call_and_return_conditional_losses_1307904

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
ож
╗
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309920
lambda_9_input;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_36_conv2d_readvariableop_resource: 7
)conv2d_36_biasadd_readvariableop_resource: C
(conv2d_37_conv2d_readvariableop_resource: А8
)conv2d_37_biasadd_readvariableop_resource:	АD
(conv2d_38_conv2d_readvariableop_resource:АА8
)conv2d_38_biasadd_readvariableop_resource:	АD
(conv2d_39_conv2d_readvariableop_resource:АА8
)conv2d_39_biasadd_readvariableop_resource:	А;
'dense_27_matmul_readvariableop_resource:
А@А7
(dense_27_biasadd_readvariableop_resource:	А;
'dense_28_matmul_readvariableop_resource:
АА7
(dense_28_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_9/AssignNewValueв&batch_normalization_9/AssignNewValue_1в5batch_normalization_9/FusedBatchNormV3/ReadVariableOpв7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_9/ReadVariableOpв&batch_normalization_9/ReadVariableOp_1в conv2d_36/BiasAdd/ReadVariableOpвconv2d_36/Conv2D/ReadVariableOpв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв conv2d_37/BiasAdd/ReadVariableOpвconv2d_37/Conv2D/ReadVariableOpв conv2d_38/BiasAdd/ReadVariableOpвconv2d_38/Conv2D/ReadVariableOpв conv2d_39/BiasAdd/ReadVariableOpвconv2d_39/Conv2D/ReadVariableOpвdense_27/BiasAdd/ReadVariableOpвdense_27/MatMul/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpвdense_28/BiasAdd/ReadVariableOpвdense_28/MatMul/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpХ
lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_9/strided_slice/stackЩ
lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_9/strided_slice/stack_1Щ
lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_9/strided_slice/stack_2▓
lambda_9/strided_sliceStridedSlicelambda_9_input%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_9/strided_slice╢
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOp╝
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1щ
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ї
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_9/FusedBatchNormV3░
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue╝
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1│
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOpх
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_36/Conv2Dк
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp░
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_36/Relu╩
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool┤
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_37/Conv2D/ReadVariableOp▌
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_37/Conv2Dл
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp▒
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_37/Relu╦
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool╡
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_38/Conv2D/ReadVariableOp▌
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_38/Conv2Dл
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_38/Relu╦
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool╡
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_39/Conv2D/ReadVariableOp▌
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
conv2d_39/Conv2Dл
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▒
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
conv2d_39/Relu╦
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPooly
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_27/dropout/Const╕
dropout_27/dropout/MulMul!max_pooling2d_39/MaxPool:output:0!dropout_27/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout_27/dropout/MulЕ
dropout_27/dropout/ShapeShape!max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shape▐
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype021
/dropout_27/dropout/random_uniform/RandomUniformЛ
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_27/dropout/GreaterEqual/yє
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2!
dropout_27/dropout/GreaterEqualй
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_27/dropout/Castп
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_27/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_9/ConstЬ
flatten_9/ReshapeReshapedropout_27/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_9/Reshapeк
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_27/MatMul/ReadVariableOpг
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_27/MatMulи
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_27/BiasAdd/ReadVariableOpж
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_27/Reluy
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_28/dropout/Constк
dropout_28/dropout/MulMuldense_27/Relu:activations:0!dropout_28/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_28/dropout/Mul
dropout_28/dropout/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
dropout_28/dropout/Shape╓
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_28/dropout/random_uniform/RandomUniformЛ
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_28/dropout/GreaterEqual/yы
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_28/dropout/GreaterEqualб
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_28/dropout/Castз
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_28/dropout/Mul_1к
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_28/MatMul/ReadVariableOpе
dense_28/MatMulMatMuldropout_28/dropout/Mul_1:z:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_28/MatMulи
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_28/BiasAdd/ReadVariableOpж
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_28/Reluy
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_29/dropout/Constк
dropout_29/dropout/MulMuldense_28/Relu:activations:0!dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_29/dropout/Mul
dropout_29/dropout/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:2
dropout_29/dropout/Shape╓
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_29/dropout/random_uniform/RandomUniformЛ
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_29/dropout/GreaterEqual/yы
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_29/dropout/GreaterEqualб
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_29/dropout/Castз
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_29/dropout/Mul_1┘
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╨
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╨
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul║
IdentityIdentitydropout_29/dropout/Mul_1:z:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2L
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
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_9_input
╠
а
+__inference_conv2d_36_layer_call_fn_1310270

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
F__inference_conv2d_36_layer_call_and_return_conditional_losses_13076942
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
М
Э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1307481

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
Ю
Б
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1310281

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
┘
G
+__inference_flatten_9_layer_call_fn_1310368

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
:         А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_13077682
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
║
н
E__inference_dense_28_layer_call_and_return_conditional_losses_1307817

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpП
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
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
й
Ъ
*__inference_dense_28_layer_call_fn_1310459

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
E__inference_dense_28_layer_call_and_return_conditional_losses_13078172
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
°
e
G__inference_dropout_28_layer_call_and_return_conditional_losses_1307798

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
╙
г
+__inference_conv2d_39_layer_call_fn_1310330

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_13077482
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
р
N
2__inference_max_pooling2d_38_layer_call_fn_1307621

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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_13076152
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
Ё
╥
7__inference_batch_normalization_9_layer_call_fn_1310199

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall╗
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
GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_13074812
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
╦
H
,__inference_dropout_29_layer_call_fn_1310481

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
G__inference_dropout_29_layer_call_and_return_conditional_losses_13078282
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
,__inference_dropout_28_layer_call_fn_1310427

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
G__inference_dropout_28_layer_call_and_return_conditional_losses_13079372
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
°
┴
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1308052

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
├
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1310096

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
╬u
Д
 __inference__traced_save_1310719
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
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop/
+savev2_conv2d_39_kernel_read_readvariableop-
)savev2_conv2d_39_bias_read_readvariableop.
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
2savev2_adam_conv2d_36_kernel_m_read_readvariableop4
0savev2_adam_conv2d_36_bias_m_read_readvariableop6
2savev2_adam_conv2d_37_kernel_m_read_readvariableop4
0savev2_adam_conv2d_37_bias_m_read_readvariableop6
2savev2_adam_conv2d_38_kernel_m_read_readvariableop4
0savev2_adam_conv2d_38_bias_m_read_readvariableop6
2savev2_adam_conv2d_39_kernel_m_read_readvariableop4
0savev2_adam_conv2d_39_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableopA
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
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop
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
ShardedFilenameъ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*№
valueЄBя<B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*Н
valueГBА<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesа
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop2savev2_adam_conv2d_36_kernel_m_read_readvariableop0savev2_adam_conv2d_36_bias_m_read_readvariableop2savev2_adam_conv2d_37_kernel_m_read_readvariableop0savev2_adam_conv2d_37_bias_m_read_readvariableop2savev2_adam_conv2d_38_kernel_m_read_readvariableop0savev2_adam_conv2d_38_bias_m_read_readvariableop2savev2_adam_conv2d_39_kernel_m_read_readvariableop0savev2_adam_conv2d_39_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop2savev2_adam_conv2d_36_kernel_v_read_readvariableop0savev2_adam_conv2d_36_bias_v_read_readvariableop2savev2_adam_conv2d_37_kernel_v_read_readvariableop0savev2_adam_conv2d_37_bias_v_read_readvariableop2savev2_adam_conv2d_38_kernel_v_read_readvariableop0savev2_adam_conv2d_38_bias_v_read_readvariableop2savev2_adam_conv2d_39_kernel_v_read_readvariableop0savev2_adam_conv2d_39_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<	2
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

identity_1Identity_1:output:0*╕
_input_shapesж
г: :	А:: : : : : ::: : : А:А:АА:А:АА:А:
А@А:А:
АА:А::: : : : :	А:::: : : А:А:АА:А:АА:А:
А@А:А:
АА:А:	А:::: : : А:А:АА:А:АА:А:
А@А:А:
АА:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	А: 
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
: А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
А@А:!
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
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	А: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
: : !

_output_shapes
: :-")
'
_output_shapes
: А:!#

_output_shapes	
:А:.$*
(
_output_shapes
:АА:!%

_output_shapes	
:А:.&*
(
_output_shapes
:АА:!'

_output_shapes	
:А:&("
 
_output_shapes
:
А@А:!)

_output_shapes	
:А:&*"
 
_output_shapes
:
АА:!+

_output_shapes	
:А:%,!

_output_shapes
:	А: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
: : 1

_output_shapes
: :-2)
'
_output_shapes
: А:!3

_output_shapes	
:А:.4*
(
_output_shapes
:АА:!5

_output_shapes	
:А:.6*
(
_output_shapes
:АА:!7

_output_shapes	
:А:&8"
 
_output_shapes
:
А@А:!9

_output_shapes	
:А:&:"
 
_output_shapes
:
АА:!;

_output_shapes	
:А:<

_output_shapes
: 
в3
Т
A__inference_CNN3_layer_call_and_return_conditional_losses_1308631

inputs"
sequential_9_1308574:"
sequential_9_1308576:"
sequential_9_1308578:"
sequential_9_1308580:.
sequential_9_1308582: "
sequential_9_1308584: /
sequential_9_1308586: А#
sequential_9_1308588:	А0
sequential_9_1308590:АА#
sequential_9_1308592:	А0
sequential_9_1308594:АА#
sequential_9_1308596:	А(
sequential_9_1308598:
А@А#
sequential_9_1308600:	А(
sequential_9_1308602:
АА#
sequential_9_1308604:	А#
dense_29_1308607:	А
dense_29_1308609:
identityИв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpв dense_29/StatefulPartitionedCallв$sequential_9/StatefulPartitionedCall 
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_1308574sequential_9_1308576sequential_9_1308578sequential_9_1308580sequential_9_1308582sequential_9_1308584sequential_9_1308586sequential_9_1308588sequential_9_1308590sequential_9_1308592sequential_9_1308594sequential_9_1308596sequential_9_1308598sequential_9_1308600sequential_9_1308602sequential_9_1308604*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_13081912&
$sequential_9/StatefulPartitionedCall├
 dense_29/StatefulPartitionedCallStatefulPartitionedCall-sequential_9/StatefulPartitionedCall:output:0dense_29_1308607dense_29_1308609*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_13084542"
 dense_29/StatefulPartitionedCall┼
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1308582*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╜
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1308598* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╜
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1308602* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulф
IdentityIdentity)dense_29/StatefulPartitionedCall:output:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp!^dense_29/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▐╨
Э
A__inference_CNN3_layer_call_and_return_conditional_losses_1309332
input_1H
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: АE
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_38_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_39_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	АH
4sequential_9_dense_27_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_27_biasadd_readvariableop_resource:	АH
4sequential_9_dense_28_matmul_readvariableop_resource:
ААD
5sequential_9_dense_28_biasadd_readvariableop_resource:	А:
'dense_29_matmul_readvariableop_resource:	А6
(dense_29_biasadd_readvariableop_resource:
identityИв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpвdense_29/BiasAdd/ReadVariableOpвdense_29/MatMul/ReadVariableOpв1sequential_9/batch_normalization_9/AssignNewValueв3sequential_9/batch_normalization_9/AssignNewValue_1вBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpвDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в1sequential_9/batch_normalization_9/ReadVariableOpв3sequential_9/batch_normalization_9/ReadVariableOp_1в-sequential_9/conv2d_36/BiasAdd/ReadVariableOpв,sequential_9/conv2d_36/Conv2D/ReadVariableOpв-sequential_9/conv2d_37/BiasAdd/ReadVariableOpв,sequential_9/conv2d_37/Conv2D/ReadVariableOpв-sequential_9/conv2d_38/BiasAdd/ReadVariableOpв,sequential_9/conv2d_38/Conv2D/ReadVariableOpв-sequential_9/conv2d_39/BiasAdd/ReadVariableOpв,sequential_9/conv2d_39/Conv2D/ReadVariableOpв,sequential_9/dense_27/BiasAdd/ReadVariableOpв+sequential_9/dense_27/MatMul/ReadVariableOpв,sequential_9/dense_28/BiasAdd/ReadVariableOpв+sequential_9/dense_28/MatMul/ReadVariableOpп
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack│
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1│
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2ь
#sequential_9/lambda_9/strided_sliceStridedSliceinput_12sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice▌
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpу
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1Р
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1╨
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<25
3sequential_9/batch_normalization_9/FusedBatchNormV3ё
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_9/batch_normalization_9/AssignNewValue¤
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_9/batch_normalization_9/AssignNewValue_1┌
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D╤
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpф
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_9/conv2d_36/BiasAddе
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_9/conv2d_36/Reluё
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool█
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D╥
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpх
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_9/conv2d_37/BiasAddж
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_9/conv2d_37/ReluЄ
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool▄
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D╥
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpх
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_9/conv2d_38/BiasAddж
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_9/conv2d_38/ReluЄ
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool▄
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D╥
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpх
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_9/conv2d_39/BiasAddж
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_9/conv2d_39/ReluЄ
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPoolУ
%sequential_9/dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2'
%sequential_9/dropout_27/dropout/Constь
#sequential_9/dropout_27/dropout/MulMul.sequential_9/max_pooling2d_39/MaxPool:output:0.sequential_9/dropout_27/dropout/Const:output:0*
T0*0
_output_shapes
:         А2%
#sequential_9/dropout_27/dropout/Mulм
%sequential_9/dropout_27/dropout/ShapeShape.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_27/dropout/ShapeЕ
<sequential_9/dropout_27/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_27/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02>
<sequential_9/dropout_27/dropout/random_uniform/RandomUniformе
.sequential_9/dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_9/dropout_27/dropout/GreaterEqual/yз
,sequential_9/dropout_27/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_27/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_27/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2.
,sequential_9/dropout_27/dropout/GreaterEqual╨
$sequential_9/dropout_27/dropout/CastCast0sequential_9/dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2&
$sequential_9/dropout_27/dropout/Castу
%sequential_9/dropout_27/dropout/Mul_1Mul'sequential_9/dropout_27/dropout/Mul:z:0(sequential_9/dropout_27/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2'
%sequential_9/dropout_27/dropout/Mul_1Н
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_9/flatten_9/Const╨
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/dropout/Mul_1:z:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_9/flatten_9/Reshape╤
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp╫
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/MatMul╧
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp┌
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/BiasAddЫ
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/ReluУ
%sequential_9/dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_28/dropout/Const▐
#sequential_9/dropout_28/dropout/MulMul(sequential_9/dense_27/Relu:activations:0.sequential_9/dropout_28/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_9/dropout_28/dropout/Mulж
%sequential_9/dropout_28/dropout/ShapeShape(sequential_9/dense_27/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_28/dropout/Shape¤
<sequential_9/dropout_28/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_28/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_9/dropout_28/dropout/random_uniform/RandomUniformе
.sequential_9/dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_28/dropout/GreaterEqual/yЯ
,sequential_9/dropout_28/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_28/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_28/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_9/dropout_28/dropout/GreaterEqual╚
$sequential_9/dropout_28/dropout/CastCast0sequential_9/dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_9/dropout_28/dropout/Cast█
%sequential_9/dropout_28/dropout/Mul_1Mul'sequential_9/dropout_28/dropout/Mul:z:0(sequential_9/dropout_28/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_9/dropout_28/dropout/Mul_1╤
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp┘
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/dropout/Mul_1:z:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/MatMul╧
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp┌
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/BiasAddЫ
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/ReluУ
%sequential_9/dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_29/dropout/Const▐
#sequential_9/dropout_29/dropout/MulMul(sequential_9/dense_28/Relu:activations:0.sequential_9/dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_9/dropout_29/dropout/Mulж
%sequential_9/dropout_29/dropout/ShapeShape(sequential_9/dense_28/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_29/dropout/Shape¤
<sequential_9/dropout_29/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_9/dropout_29/dropout/random_uniform/RandomUniformе
.sequential_9/dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_29/dropout/GreaterEqual/yЯ
,sequential_9/dropout_29/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_29/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_9/dropout_29/dropout/GreaterEqual╚
$sequential_9/dropout_29/dropout/CastCast0sequential_9/dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_9/dropout_29/dropout/Cast█
%sequential_9/dropout_29/dropout/Mul_1Mul'sequential_9/dropout_29/dropout/Mul:z:0(sequential_9/dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_9/dropout_29/dropout/Mul_1й
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_29/MatMul/ReadVariableOp▒
dense_29/MatMulMatMul)sequential_9/dropout_29/dropout/Mul_1:z:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/MatMulз
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpе
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_29/Softmaxц
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul▌
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul▌
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulф	
IdentityIdentitydense_29/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_12И
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2М
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
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
р
N
2__inference_max_pooling2d_36_layer_call_fn_1307597

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
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_13075912
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
├
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1308079

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
■Б
ы
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309808
lambda_9_input;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_36_conv2d_readvariableop_resource: 7
)conv2d_36_biasadd_readvariableop_resource: C
(conv2d_37_conv2d_readvariableop_resource: А8
)conv2d_37_biasadd_readvariableop_resource:	АD
(conv2d_38_conv2d_readvariableop_resource:АА8
)conv2d_38_biasadd_readvariableop_resource:	АD
(conv2d_39_conv2d_readvariableop_resource:АА8
)conv2d_39_biasadd_readvariableop_resource:	А;
'dense_27_matmul_readvariableop_resource:
А@А7
(dense_27_biasadd_readvariableop_resource:	А;
'dense_28_matmul_readvariableop_resource:
АА7
(dense_28_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_9/FusedBatchNormV3/ReadVariableOpв7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_9/ReadVariableOpв&batch_normalization_9/ReadVariableOp_1в conv2d_36/BiasAdd/ReadVariableOpвconv2d_36/Conv2D/ReadVariableOpв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв conv2d_37/BiasAdd/ReadVariableOpвconv2d_37/Conv2D/ReadVariableOpв conv2d_38/BiasAdd/ReadVariableOpвconv2d_38/Conv2D/ReadVariableOpв conv2d_39/BiasAdd/ReadVariableOpвconv2d_39/Conv2D/ReadVariableOpвdense_27/BiasAdd/ReadVariableOpвdense_27/MatMul/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpвdense_28/BiasAdd/ReadVariableOpвdense_28/MatMul/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpХ
lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_9/strided_slice/stackЩ
lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_9/strided_slice/stack_1Щ
lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_9/strided_slice/stack_2▓
lambda_9/strided_sliceStridedSlicelambda_9_input%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_9/strided_slice╢
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOp╝
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1щ
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3│
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOpх
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_36/Conv2Dк
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp░
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_36/Relu╩
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool┤
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_37/Conv2D/ReadVariableOp▌
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_37/Conv2Dл
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp▒
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_37/Relu╦
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool╡
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_38/Conv2D/ReadVariableOp▌
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_38/Conv2Dл
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_38/Relu╦
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool╡
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_39/Conv2D/ReadVariableOp▌
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
conv2d_39/Conv2Dл
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▒
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
conv2d_39/Relu╦
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPoolФ
dropout_27/IdentityIdentity!max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         А2
dropout_27/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_9/ConstЬ
flatten_9/ReshapeReshapedropout_27/Identity:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_9/Reshapeк
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_27/MatMul/ReadVariableOpг
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_27/MatMulи
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_27/BiasAdd/ReadVariableOpж
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_27/ReluЖ
dropout_28/IdentityIdentitydense_27/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_28/Identityк
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_28/MatMul/ReadVariableOpе
dense_28/MatMulMatMuldropout_28/Identity:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_28/MatMulи
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_28/BiasAdd/ReadVariableOpж
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_28/ReluЖ
dropout_29/IdentityIdentitydense_28/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_29/Identity┘
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╨
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╨
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulъ
IdentityIdentitydropout_29/Identity:output:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2n
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
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_9_input
щ
┤
__inference_loss_fn_2_1310519N
:dense_28_kernel_regularizer_square_readvariableop_resource:
АА
identityИв1dense_28/kernel/Regularizer/Square/ReadVariableOpу
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_28_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulЪ
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
В
Є
&__inference_CNN3_layer_call_fn_1309373
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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:	А

unknown_16:
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *'
_output_shapes
:         *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_13084792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
Ы
╝
__inference_loss_fn_0_1310497U
;conv2d_36_kernel_regularizer_square_readvariableop_resource: 
identityИв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpь
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_36_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulЬ
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
ъ
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1310363

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
║
н
E__inference_dense_27_layer_call_and_return_conditional_losses_1310391

inputs2
matmul_readvariableop_resource:
А@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
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
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
║В
ь
__inference_call_1249596

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: АE
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_38_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_39_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	АH
4sequential_9_dense_27_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_27_biasadd_readvariableop_resource:	АH
4sequential_9_dense_28_matmul_readvariableop_resource:
ААD
5sequential_9_dense_28_biasadd_readvariableop_resource:	А:
'dense_29_matmul_readvariableop_resource:	А6
(dense_29_biasadd_readvariableop_resource:
identityИвdense_29/BiasAdd/ReadVariableOpвdense_29/MatMul/ReadVariableOpвBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpвDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в1sequential_9/batch_normalization_9/ReadVariableOpв3sequential_9/batch_normalization_9/ReadVariableOp_1в-sequential_9/conv2d_36/BiasAdd/ReadVariableOpв,sequential_9/conv2d_36/Conv2D/ReadVariableOpв-sequential_9/conv2d_37/BiasAdd/ReadVariableOpв,sequential_9/conv2d_37/Conv2D/ReadVariableOpв-sequential_9/conv2d_38/BiasAdd/ReadVariableOpв,sequential_9/conv2d_38/Conv2D/ReadVariableOpв-sequential_9/conv2d_39/BiasAdd/ReadVariableOpв,sequential_9/conv2d_39/Conv2D/ReadVariableOpв,sequential_9/dense_27/BiasAdd/ReadVariableOpв+sequential_9/dense_27/MatMul/ReadVariableOpв,sequential_9/dense_28/BiasAdd/ReadVariableOpв+sequential_9/dense_28/MatMul/ReadVariableOpп
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack│
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1│
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2у
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice▌
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpу
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1Р
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1║
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3┌
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpС
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D╤
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp▄
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2 
sequential_9/conv2d_36/BiasAddЭ
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_9/conv2d_36/Reluщ
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool█
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpЙ
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D╥
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp▌
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2 
sequential_9/conv2d_37/BiasAddЮ
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_9/conv2d_37/Reluъ
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool▄
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpЙ
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D╥
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp▌
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_9/conv2d_38/BiasAddЮ
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_9/conv2d_38/Reluъ
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool▄
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpЙ
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D╥
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp▌
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А2 
sequential_9/conv2d_39/BiasAddЮ
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*(
_output_shapes
:А		А2
sequential_9/conv2d_39/Reluъ
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool│
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*(
_output_shapes
:АА2"
 sequential_9/dropout_27/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_9/flatten_9/Const╚
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0* 
_output_shapes
:
АА@2 
sequential_9/flatten_9/Reshape╤
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp╧
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_27/MatMul╧
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp╥
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_27/BiasAddУ
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_27/Reluе
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_9/dropout_28/Identity╤
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp╤
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_28/MatMul╧
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp╥
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_28/BiasAddУ
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_28/Reluе
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_9/dropout_29/Identityй
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_29/MatMul/ReadVariableOpй
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_29/MatMulз
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpЭ
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_29/BiasAddt
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_29/Softmax╒
IdentityIdentitydense_29/Softmax:softmax:0 ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:АKK: : : : : : : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2И
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2М
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
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
╣

ў
E__inference_dense_29_layer_call_and_return_conditional_losses_1310079

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
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
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
н
i
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1307591

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
╨
╚
.__inference_sequential_9_layer_call_fn_1309957
lambda_9_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А
identityИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCalllambda_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *(
_output_shapes
:         А*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_13078492
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_9_input
й
Ъ
*__inference_dense_27_layer_call_fn_1310400

inputs
unknown:
А@А
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
E__inference_dense_27_layer_call_and_return_conditional_losses_13077872
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
р
N
2__inference_max_pooling2d_39_layer_call_fn_1307633

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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_13076272
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
в
В
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1310301

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
А
Є
&__inference_CNN3_layer_call_fn_1309496
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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:	А

unknown_16:
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_13086312
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
н
i
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1307615

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
G__inference_dropout_28_layer_call_and_return_conditional_losses_1310417

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
┌
д
"__inference__wrapped_model_1307459
input_1
cnn3_1307421:
cnn3_1307423:
cnn3_1307425:
cnn3_1307427:&
cnn3_1307429: 
cnn3_1307431: '
cnn3_1307433: А
cnn3_1307435:	А(
cnn3_1307437:АА
cnn3_1307439:	А(
cnn3_1307441:АА
cnn3_1307443:	А 
cnn3_1307445:
А@А
cnn3_1307447:	А 
cnn3_1307449:
АА
cnn3_1307451:	А
cnn3_1307453:	А
cnn3_1307455:
identityИвCNN3/StatefulPartitionedCallр
CNN3/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn3_1307421cnn3_1307423cnn3_1307425cnn3_1307427cnn3_1307429cnn3_1307431cnn3_1307433cnn3_1307435cnn3_1307437cnn3_1307439cnn3_1307441cnn3_1307443cnn3_1307445cnn3_1307447cnn3_1307449cnn3_1307451cnn3_1307453cnn3_1307455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *!
fR
__inference_call_12473602
CNN3/StatefulPartitionedCallШ
IdentityIdentity%CNN3/StatefulPartitionedCall:output:0^CNN3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2<
CNN3/StatefulPartitionedCallCNN3/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
т
ё
%__inference_signature_wrapper_1308898
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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:	А

unknown_16:
identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *'
_output_shapes
:         *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_13074592
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
╢
f
G__inference_dropout_29_layer_call_and_return_conditional_losses_1310476

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
тг
▓
A__inference_CNN3_layer_call_and_return_conditional_losses_1308996

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: АE
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_38_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_39_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	АH
4sequential_9_dense_27_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_27_biasadd_readvariableop_resource:	АH
4sequential_9_dense_28_matmul_readvariableop_resource:
ААD
5sequential_9_dense_28_biasadd_readvariableop_resource:	А:
'dense_29_matmul_readvariableop_resource:	А6
(dense_29_biasadd_readvariableop_resource:
identityИв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpвdense_29/BiasAdd/ReadVariableOpвdense_29/MatMul/ReadVariableOpвBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpвDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в1sequential_9/batch_normalization_9/ReadVariableOpв3sequential_9/batch_normalization_9/ReadVariableOp_1в-sequential_9/conv2d_36/BiasAdd/ReadVariableOpв,sequential_9/conv2d_36/Conv2D/ReadVariableOpв-sequential_9/conv2d_37/BiasAdd/ReadVariableOpв,sequential_9/conv2d_37/Conv2D/ReadVariableOpв-sequential_9/conv2d_38/BiasAdd/ReadVariableOpв,sequential_9/conv2d_38/Conv2D/ReadVariableOpв-sequential_9/conv2d_39/BiasAdd/ReadVariableOpв,sequential_9/conv2d_39/Conv2D/ReadVariableOpв,sequential_9/dense_27/BiasAdd/ReadVariableOpв+sequential_9/dense_27/MatMul/ReadVariableOpв,sequential_9/dense_28/BiasAdd/ReadVariableOpв+sequential_9/dense_28/MatMul/ReadVariableOpп
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack│
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1│
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2ы
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice▌
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpу
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1Р
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3┌
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D╤
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpф
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_9/conv2d_36/BiasAddе
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_9/conv2d_36/Reluё
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool█
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D╥
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpх
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_9/conv2d_37/BiasAddж
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_9/conv2d_37/ReluЄ
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool▄
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D╥
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpх
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_9/conv2d_38/BiasAddж
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_9/conv2d_38/ReluЄ
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool▄
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D╥
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpх
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_9/conv2d_39/BiasAddж
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_9/conv2d_39/ReluЄ
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool╗
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_9/dropout_27/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_9/flatten_9/Const╨
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_9/flatten_9/Reshape╤
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp╫
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/MatMul╧
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp┌
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/BiasAddЫ
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/Reluн
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_9/dropout_28/Identity╤
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp┘
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/MatMul╧
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp┌
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/BiasAddЫ
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/Reluн
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_9/dropout_29/Identityй
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_29/MatMul/ReadVariableOp▒
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/MatMulз
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpе
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_29/Softmaxц
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul▌
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul▌
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul·
IdentityIdentitydense_29/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2И
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2М
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
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Жe
Н	
I__inference_sequential_9_layer_call_and_return_conditional_losses_1307849

inputs+
batch_normalization_9_1307668:+
batch_normalization_9_1307670:+
batch_normalization_9_1307672:+
batch_normalization_9_1307674:+
conv2d_36_1307695: 
conv2d_36_1307697: ,
conv2d_37_1307713: А 
conv2d_37_1307715:	А-
conv2d_38_1307731:АА 
conv2d_38_1307733:	А-
conv2d_39_1307749:АА 
conv2d_39_1307751:	А$
dense_27_1307788:
А@А
dense_27_1307790:	А$
dense_28_1307818:
АА
dense_28_1307820:	А
identityИв-batch_normalization_9/StatefulPartitionedCallв!conv2d_36/StatefulPartitionedCallв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв!conv2d_37/StatefulPartitionedCallв!conv2d_38/StatefulPartitionedCallв!conv2d_39/StatefulPartitionedCallв dense_27/StatefulPartitionedCallв1dense_27/kernel/Regularizer/Square/ReadVariableOpв dense_28/StatefulPartitionedCallв1dense_28/kernel/Regularizer/Square/ReadVariableOpт
lambda_9/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8В *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_13076482
lambda_9/PartitionedCall┬
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall!lambda_9/PartitionedCall:output:0batch_normalization_9_1307668batch_normalization_9_1307670batch_normalization_9_1307672batch_normalization_9_1307674*
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
GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_13076672/
-batch_normalization_9/StatefulPartitionedCall┘
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_36_1307695conv2d_36_1307697*
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
F__inference_conv2d_36_layer_call_and_return_conditional_losses_13076942#
!conv2d_36/StatefulPartitionedCallЮ
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_13075912"
 max_pooling2d_36/PartitionedCall═
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_37_1307713conv2d_37_1307715*
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
F__inference_conv2d_37_layer_call_and_return_conditional_losses_13077122#
!conv2d_37/StatefulPartitionedCallЯ
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_13076032"
 max_pooling2d_37/PartitionedCall═
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_38_1307731conv2d_38_1307733*
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
F__inference_conv2d_38_layer_call_and_return_conditional_losses_13077302#
!conv2d_38/StatefulPartitionedCallЯ
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_13076152"
 max_pooling2d_38/PartitionedCall═
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_1307749conv2d_39_1307751*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_13077482#
!conv2d_39/StatefulPartitionedCallЯ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_13076272"
 max_pooling2d_39/PartitionedCallМ
dropout_27/PartitionedCallPartitionedCall)max_pooling2d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_27_layer_call_and_return_conditional_losses_13077602
dropout_27/PartitionedCall√
flatten_9/PartitionedCallPartitionedCall#dropout_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_13077682
flatten_9/PartitionedCall╣
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_27_1307788dense_27_1307790*
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
E__inference_dense_27_layer_call_and_return_conditional_losses_13077872"
 dense_27/StatefulPartitionedCallД
dropout_28/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
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
G__inference_dropout_28_layer_call_and_return_conditional_losses_13077982
dropout_28/PartitionedCall║
 dense_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0dense_28_1307818dense_28_1307820*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_13078172"
 dense_28/StatefulPartitionedCallД
dropout_29/PartitionedCallPartitionedCall)dense_28/StatefulPartitionedCall:output:0*
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
G__inference_dropout_29_layer_call_and_return_conditional_losses_13078282
dropout_29/PartitionedCall┬
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_36_1307695*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╣
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_27_1307788* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╣
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_28_1307818* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulЫ
IdentityIdentity#dropout_29/PartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall2^dense_27/kernel/Regularizer/Square/ReadVariableOp!^dense_28/StatefulPartitionedCall2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
н
i
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1307627

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
М
Э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310132

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
ў
f
G__inference_dropout_27_layer_call_and_return_conditional_losses_1307976

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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ў
f
G__inference_dropout_27_layer_call_and_return_conditional_losses_1310347

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
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         А*
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
:         А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
в
В
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1307748

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
в
В
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1307730

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
├
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1307648

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
╙
г
+__inference_conv2d_38_layer_call_fn_1310310

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
F__inference_conv2d_38_layer_call_and_return_conditional_losses_13077302
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
─
Э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1307667

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
└
┴
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310150

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
,__inference_dropout_28_layer_call_fn_1310422

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
G__inference_dropout_28_layer_call_and_return_conditional_losses_13077982
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
у
F
*__inference_lambda_9_layer_call_fn_1310114

inputs
identity╨
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
GPU2 *0J 8В *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_13080792
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
Ю
Б
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1307712

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
°
e
G__inference_dropout_28_layer_call_and_return_conditional_losses_1310405

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
ю
╥
7__inference_batch_normalization_9_layer_call_fn_1310212

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall╣
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
GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_13075252
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
║В
ь
__inference_call_1249516

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: АE
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_38_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_39_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	АH
4sequential_9_dense_27_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_27_biasadd_readvariableop_resource:	АH
4sequential_9_dense_28_matmul_readvariableop_resource:
ААD
5sequential_9_dense_28_biasadd_readvariableop_resource:	А:
'dense_29_matmul_readvariableop_resource:	А6
(dense_29_biasadd_readvariableop_resource:
identityИвdense_29/BiasAdd/ReadVariableOpвdense_29/MatMul/ReadVariableOpвBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpвDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в1sequential_9/batch_normalization_9/ReadVariableOpв3sequential_9/batch_normalization_9/ReadVariableOp_1в-sequential_9/conv2d_36/BiasAdd/ReadVariableOpв,sequential_9/conv2d_36/Conv2D/ReadVariableOpв-sequential_9/conv2d_37/BiasAdd/ReadVariableOpв,sequential_9/conv2d_37/Conv2D/ReadVariableOpв-sequential_9/conv2d_38/BiasAdd/ReadVariableOpв,sequential_9/conv2d_38/Conv2D/ReadVariableOpв-sequential_9/conv2d_39/BiasAdd/ReadVariableOpв,sequential_9/conv2d_39/Conv2D/ReadVariableOpв,sequential_9/dense_27/BiasAdd/ReadVariableOpв+sequential_9/dense_27/MatMul/ReadVariableOpв,sequential_9/dense_28/BiasAdd/ReadVariableOpв+sequential_9/dense_28/MatMul/ReadVariableOpп
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack│
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1│
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2у
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice▌
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpу
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1Р
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1║
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3┌
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpС
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D╤
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp▄
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2 
sequential_9/conv2d_36/BiasAddЭ
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_9/conv2d_36/Reluщ
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool█
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpЙ
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D╥
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOp▌
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2 
sequential_9/conv2d_37/BiasAddЮ
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_9/conv2d_37/Reluъ
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool▄
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpЙ
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D╥
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOp▌
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_9/conv2d_38/BiasAddЮ
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_9/conv2d_38/Reluъ
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool▄
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpЙ
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D╥
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOp▌
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А2 
sequential_9/conv2d_39/BiasAddЮ
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*(
_output_shapes
:А		А2
sequential_9/conv2d_39/Reluъ
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool│
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*(
_output_shapes
:АА2"
 sequential_9/dropout_27/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_9/flatten_9/Const╚
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0* 
_output_shapes
:
АА@2 
sequential_9/flatten_9/Reshape╤
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp╧
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_27/MatMul╧
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp╥
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_27/BiasAddУ
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_27/Reluе
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_9/dropout_28/Identity╤
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp╤
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_28/MatMul╧
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp╥
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_28/BiasAddУ
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_28/Reluе
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_9/dropout_29/Identityй
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_29/MatMul/ReadVariableOpй
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_29/MatMulз
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpЭ
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_29/BiasAddt
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_29/Softmax╒
IdentityIdentitydense_29/Softmax:softmax:0 ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:АKK: : : : : : : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2И
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2М
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
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
ы
H
,__inference_dropout_27_layer_call_fn_1310352

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
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_27_layer_call_and_return_conditional_losses_13077602
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╫
e
,__inference_dropout_29_layer_call_fn_1310486

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
G__inference_dropout_29_layer_call_and_return_conditional_losses_13079042
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
 
ё
&__inference_CNN3_layer_call_fn_1309414

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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:	А

unknown_16:
identityИвStatefulPartitionedCall═
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
 *'
_output_shapes
:         *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_13084792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

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
цБ
у
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309605

inputs;
-batch_normalization_9_readvariableop_resource:=
/batch_normalization_9_readvariableop_1_resource:L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_36_conv2d_readvariableop_resource: 7
)conv2d_36_biasadd_readvariableop_resource: C
(conv2d_37_conv2d_readvariableop_resource: А8
)conv2d_37_biasadd_readvariableop_resource:	АD
(conv2d_38_conv2d_readvariableop_resource:АА8
)conv2d_38_biasadd_readvariableop_resource:	АD
(conv2d_39_conv2d_readvariableop_resource:АА8
)conv2d_39_biasadd_readvariableop_resource:	А;
'dense_27_matmul_readvariableop_resource:
А@А7
(dense_27_biasadd_readvariableop_resource:	А;
'dense_28_matmul_readvariableop_resource:
АА7
(dense_28_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_9/FusedBatchNormV3/ReadVariableOpв7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_9/ReadVariableOpв&batch_normalization_9/ReadVariableOp_1в conv2d_36/BiasAdd/ReadVariableOpвconv2d_36/Conv2D/ReadVariableOpв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв conv2d_37/BiasAdd/ReadVariableOpвconv2d_37/Conv2D/ReadVariableOpв conv2d_38/BiasAdd/ReadVariableOpвconv2d_38/Conv2D/ReadVariableOpв conv2d_39/BiasAdd/ReadVariableOpвconv2d_39/Conv2D/ReadVariableOpвdense_27/BiasAdd/ReadVariableOpвdense_27/MatMul/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpвdense_28/BiasAdd/ReadVariableOpвdense_28/MatMul/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpХ
lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_9/strided_slice/stackЩ
lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_9/strided_slice/stack_1Щ
lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_9/strided_slice/stack_2к
lambda_9/strided_sliceStridedSliceinputs%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_9/strided_slice╢
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOp╝
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1щ
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3│
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOpх
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_36/Conv2Dк
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp░
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_36/Relu╩
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool┤
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_37/Conv2D/ReadVariableOp▌
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_37/Conv2Dл
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp▒
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_37/Relu╦
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool╡
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_38/Conv2D/ReadVariableOp▌
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_38/Conv2Dл
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_38/Relu╦
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool╡
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_39/Conv2D/ReadVariableOp▌
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
conv2d_39/Conv2Dл
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp▒
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
conv2d_39/Relu╦
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPoolФ
dropout_27/IdentityIdentity!max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         А2
dropout_27/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_9/ConstЬ
flatten_9/ReshapeReshapedropout_27/Identity:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_9/Reshapeк
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_27/MatMul/ReadVariableOpг
dense_27/MatMulMatMulflatten_9/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_27/MatMulи
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_27/BiasAdd/ReadVariableOpж
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_27/ReluЖ
dropout_28/IdentityIdentitydense_27/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_28/Identityк
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_28/MatMul/ReadVariableOpе
dense_28/MatMulMatMuldropout_28/Identity:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_28/MatMulи
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_28/BiasAdd/ReadVariableOpж
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_28/ReluЖ
dropout_29/IdentityIdentitydense_28/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_29/Identity┘
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╨
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╨
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulъ
IdentityIdentitydropout_29/Identity:output:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2n
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
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
°
e
G__inference_dropout_29_layer_call_and_return_conditional_losses_1310464

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
╢
└
.__inference_sequential_9_layer_call_fn_1310031

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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А
identityИвStatefulPartitionedCall╕
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
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_13081912
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ъ
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1307768

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
у
F
*__inference_lambda_9_layer_call_fn_1310109

inputs
identity╨
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
GPU2 *0J 8В *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_13076482
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
└
┤
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1307694

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpХ
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
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╩Д
ь
__inference_call_1247360

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: АE
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_38_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_39_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	АH
4sequential_9_dense_27_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_27_biasadd_readvariableop_resource:	АH
4sequential_9_dense_28_matmul_readvariableop_resource:
ААD
5sequential_9_dense_28_biasadd_readvariableop_resource:	А:
'dense_29_matmul_readvariableop_resource:	А6
(dense_29_biasadd_readvariableop_resource:
identityИвdense_29/BiasAdd/ReadVariableOpвdense_29/MatMul/ReadVariableOpвBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpвDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в1sequential_9/batch_normalization_9/ReadVariableOpв3sequential_9/batch_normalization_9/ReadVariableOp_1в-sequential_9/conv2d_36/BiasAdd/ReadVariableOpв,sequential_9/conv2d_36/Conv2D/ReadVariableOpв-sequential_9/conv2d_37/BiasAdd/ReadVariableOpв,sequential_9/conv2d_37/Conv2D/ReadVariableOpв-sequential_9/conv2d_38/BiasAdd/ReadVariableOpв,sequential_9/conv2d_38/Conv2D/ReadVariableOpв-sequential_9/conv2d_39/BiasAdd/ReadVariableOpв,sequential_9/conv2d_39/Conv2D/ReadVariableOpв,sequential_9/dense_27/BiasAdd/ReadVariableOpв+sequential_9/dense_27/MatMul/ReadVariableOpв,sequential_9/dense_28/BiasAdd/ReadVariableOpв+sequential_9/dense_28/MatMul/ReadVariableOpп
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack│
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1│
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2ы
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice▌
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpу
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1Р
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3┌
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D╤
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpф
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_9/conv2d_36/BiasAddе
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_9/conv2d_36/Reluё
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool█
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D╥
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpх
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_9/conv2d_37/BiasAddж
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_9/conv2d_37/ReluЄ
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool▄
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D╥
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpх
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_9/conv2d_38/BiasAddж
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_9/conv2d_38/ReluЄ
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool▄
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D╥
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpх
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_9/conv2d_39/BiasAddж
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_9/conv2d_39/ReluЄ
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool╗
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_9/dropout_27/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_9/flatten_9/Const╨
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_9/flatten_9/Reshape╤
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp╫
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/MatMul╧
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp┌
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/BiasAddЫ
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/Reluн
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_9/dropout_28/Identity╤
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp┘
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/MatMul╧
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp┌
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/BiasAddЫ
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/Reluн
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_9/dropout_29/Identityй
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_29/MatMul/ReadVariableOp▒
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/MatMulз
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpе
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_29/Softmax▌
IdentityIdentitydense_29/Softmax:softmax:0 ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2И
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2М
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
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
║
н
E__inference_dense_27_layer_call_and_return_conditional_losses_1307787

inputs2
matmul_readvariableop_resource:
А@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_27/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
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
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_27/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
└
┴
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1307525

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
║
н
E__inference_dense_28_layer_call_and_return_conditional_losses_1310450

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_28/kernel/Regularizer/Square/ReadVariableOpП
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
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_28/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
°
e
G__inference_dropout_29_layer_call_and_return_conditional_losses_1307828

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
╣

ў
E__inference_dense_29_layer_call_and_return_conditional_losses_1308454

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
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
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╬
╚
.__inference_sequential_9_layer_call_fn_1310068
lambda_9_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:
АА

unknown_14:	А
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCalllambda_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_13081912
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_9_input
╩Д
ь
__inference_call_1249676

inputsH
:sequential_9_batch_normalization_9_readvariableop_resource:J
<sequential_9_batch_normalization_9_readvariableop_1_resource:Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_36_conv2d_readvariableop_resource: D
6sequential_9_conv2d_36_biasadd_readvariableop_resource: P
5sequential_9_conv2d_37_conv2d_readvariableop_resource: АE
6sequential_9_conv2d_37_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_38_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_38_biasadd_readvariableop_resource:	АQ
5sequential_9_conv2d_39_conv2d_readvariableop_resource:ААE
6sequential_9_conv2d_39_biasadd_readvariableop_resource:	АH
4sequential_9_dense_27_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_27_biasadd_readvariableop_resource:	АH
4sequential_9_dense_28_matmul_readvariableop_resource:
ААD
5sequential_9_dense_28_biasadd_readvariableop_resource:	А:
'dense_29_matmul_readvariableop_resource:	А6
(dense_29_biasadd_readvariableop_resource:
identityИвdense_29/BiasAdd/ReadVariableOpвdense_29/MatMul/ReadVariableOpвBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpвDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1в1sequential_9/batch_normalization_9/ReadVariableOpв3sequential_9/batch_normalization_9/ReadVariableOp_1в-sequential_9/conv2d_36/BiasAdd/ReadVariableOpв,sequential_9/conv2d_36/Conv2D/ReadVariableOpв-sequential_9/conv2d_37/BiasAdd/ReadVariableOpв,sequential_9/conv2d_37/Conv2D/ReadVariableOpв-sequential_9/conv2d_38/BiasAdd/ReadVariableOpв,sequential_9/conv2d_38/Conv2D/ReadVariableOpв-sequential_9/conv2d_39/BiasAdd/ReadVariableOpв,sequential_9/conv2d_39/Conv2D/ReadVariableOpв,sequential_9/dense_27/BiasAdd/ReadVariableOpв+sequential_9/dense_27/MatMul/ReadVariableOpв,sequential_9/dense_28/BiasAdd/ReadVariableOpв+sequential_9/dense_28/MatMul/ReadVariableOpп
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack│
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1│
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2ы
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_slice▌
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpу
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1Р
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3┌
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D╤
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpф
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_9/conv2d_36/BiasAddе
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_9/conv2d_36/Reluё
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPool█
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D╥
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpх
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_9/conv2d_37/BiasAddж
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_9/conv2d_37/ReluЄ
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool▄
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D╥
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpх
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_9/conv2d_38/BiasAddж
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_9/conv2d_38/ReluЄ
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool▄
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D╥
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpх
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_9/conv2d_39/BiasAddж
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_9/conv2d_39/ReluЄ
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool╗
 sequential_9/dropout_27/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_9/dropout_27/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_9/flatten_9/Const╨
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_27/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_9/flatten_9/Reshape╤
+sequential_9/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_27/MatMul/ReadVariableOp╫
sequential_9/dense_27/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/MatMul╧
,sequential_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_27/BiasAdd/ReadVariableOp┌
sequential_9/dense_27/BiasAddBiasAdd&sequential_9/dense_27/MatMul:product:04sequential_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/BiasAddЫ
sequential_9/dense_27/ReluRelu&sequential_9/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_27/Reluн
 sequential_9/dropout_28/IdentityIdentity(sequential_9/dense_27/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_9/dropout_28/Identity╤
+sequential_9/dense_28/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_9/dense_28/MatMul/ReadVariableOp┘
sequential_9/dense_28/MatMulMatMul)sequential_9/dropout_28/Identity:output:03sequential_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/MatMul╧
,sequential_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_28/BiasAdd/ReadVariableOp┌
sequential_9/dense_28/BiasAddBiasAdd&sequential_9/dense_28/MatMul:product:04sequential_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/BiasAddЫ
sequential_9/dense_28/ReluRelu&sequential_9/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_9/dense_28/Reluн
 sequential_9/dropout_29/IdentityIdentity(sequential_9/dense_28/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_9/dropout_29/Identityй
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_29/MatMul/ReadVariableOp▒
dense_29/MatMulMatMul)sequential_9/dropout_29/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/MatMulз
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpе
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_29/BiasAdd|
dense_29/SoftmaxSoftmaxdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_29/Softmax▌
IdentityIdentitydense_29/Softmax:softmax:0 ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_27/BiasAdd/ReadVariableOp,^sequential_9/dense_27/MatMul/ReadVariableOp-^sequential_9/dense_28/BiasAdd/ReadVariableOp,^sequential_9/dense_28/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         KK: : : : : : : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2И
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2М
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
,sequential_9/dense_27/BiasAdd/ReadVariableOp,sequential_9/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_27/MatMul/ReadVariableOp+sequential_9/dense_27/MatMul/ReadVariableOp2\
,sequential_9/dense_28/BiasAdd/ReadVariableOp,sequential_9/dense_28/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_28/MatMul/ReadVariableOp+sequential_9/dense_28/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╨
в
+__inference_conv2d_37_layer_call_fn_1310290

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
F__inference_conv2d_37_layer_call_and_return_conditional_losses_13077122
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
ъi
№	
I__inference_sequential_9_layer_call_and_return_conditional_losses_1308191

inputs+
batch_normalization_9_1308125:+
batch_normalization_9_1308127:+
batch_normalization_9_1308129:+
batch_normalization_9_1308131:+
conv2d_36_1308134: 
conv2d_36_1308136: ,
conv2d_37_1308140: А 
conv2d_37_1308142:	А-
conv2d_38_1308146:АА 
conv2d_38_1308148:	А-
conv2d_39_1308152:АА 
conv2d_39_1308154:	А$
dense_27_1308160:
А@А
dense_27_1308162:	А$
dense_28_1308166:
АА
dense_28_1308168:	А
identityИв-batch_normalization_9/StatefulPartitionedCallв!conv2d_36/StatefulPartitionedCallв2conv2d_36/kernel/Regularizer/Square/ReadVariableOpв!conv2d_37/StatefulPartitionedCallв!conv2d_38/StatefulPartitionedCallв!conv2d_39/StatefulPartitionedCallв dense_27/StatefulPartitionedCallв1dense_27/kernel/Regularizer/Square/ReadVariableOpв dense_28/StatefulPartitionedCallв1dense_28/kernel/Regularizer/Square/ReadVariableOpв"dropout_27/StatefulPartitionedCallв"dropout_28/StatefulPartitionedCallв"dropout_29/StatefulPartitionedCallт
lambda_9/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8В *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_13080792
lambda_9/PartitionedCall└
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall!lambda_9/PartitionedCall:output:0batch_normalization_9_1308125batch_normalization_9_1308127batch_normalization_9_1308129batch_normalization_9_1308131*
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
GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_13080522/
-batch_normalization_9/StatefulPartitionedCall┘
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_36_1308134conv2d_36_1308136*
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
F__inference_conv2d_36_layer_call_and_return_conditional_losses_13076942#
!conv2d_36/StatefulPartitionedCallЮ
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_13075912"
 max_pooling2d_36/PartitionedCall═
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_37_1308140conv2d_37_1308142*
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
F__inference_conv2d_37_layer_call_and_return_conditional_losses_13077122#
!conv2d_37/StatefulPartitionedCallЯ
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_13076032"
 max_pooling2d_37/PartitionedCall═
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_38_1308146conv2d_38_1308148*
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
F__inference_conv2d_38_layer_call_and_return_conditional_losses_13077302#
!conv2d_38/StatefulPartitionedCallЯ
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_13076152"
 max_pooling2d_38/PartitionedCall═
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_1308152conv2d_39_1308154*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_13077482#
!conv2d_39/StatefulPartitionedCallЯ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_13076272"
 max_pooling2d_39/PartitionedCallд
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_27_layer_call_and_return_conditional_losses_13079762$
"dropout_27/StatefulPartitionedCallГ
flatten_9/PartitionedCallPartitionedCall+dropout_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_13077682
flatten_9/PartitionedCall╣
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_27_1308160dense_27_1308162*
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
E__inference_dense_27_layer_call_and_return_conditional_losses_13077872"
 dense_27/StatefulPartitionedCall┴
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0#^dropout_27/StatefulPartitionedCall*
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
G__inference_dropout_28_layer_call_and_return_conditional_losses_13079372$
"dropout_28/StatefulPartitionedCall┬
 dense_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0dense_28_1308166dense_28_1308168*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_13078172"
 dense_28/StatefulPartitionedCall┴
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
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
G__inference_dropout_29_layer_call_and_return_conditional_losses_13079042$
"dropout_29/StatefulPartitionedCall┬
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_36_1308134*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Squareб
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const┬
 conv2d_36/kernel/Regularizer/SumSum'conv2d_36/kernel/Regularizer/Square:y:0+conv2d_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/SumН
"conv2d_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_36/kernel/Regularizer/mul/x─
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul╣
1dense_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_27_1308160* 
_output_shapes
:
А@А*
dtype023
1dense_27/kernel/Regularizer/Square/ReadVariableOp╕
"dense_27/kernel/Regularizer/SquareSquare9dense_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_27/kernel/Regularizer/SquareЧ
!dense_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_27/kernel/Regularizer/Const╛
dense_27/kernel/Regularizer/SumSum&dense_27/kernel/Regularizer/Square:y:0*dense_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/SumЛ
!dense_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_27/kernel/Regularizer/mul/x└
dense_27/kernel/Regularizer/mulMul*dense_27/kernel/Regularizer/mul/x:output:0(dense_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_27/kernel/Regularizer/mul╣
1dense_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_28_1308166* 
_output_shapes
:
АА*
dtype023
1dense_28/kernel/Regularizer/Square/ReadVariableOp╕
"dense_28/kernel/Regularizer/SquareSquare9dense_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_28/kernel/Regularizer/SquareЧ
!dense_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_28/kernel/Regularizer/Const╛
dense_28/kernel/Regularizer/SumSum&dense_28/kernel/Regularizer/Square:y:0*dense_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/SumЛ
!dense_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_28/kernel/Regularizer/mul/x└
dense_28/kernel/Regularizer/mulMul*dense_28/kernel/Regularizer/mul/x:output:0(dense_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_28/kernel/Regularizer/mulТ
IdentityIdentity+dropout_29/StatefulPartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall2^dense_27/kernel/Regularizer/Square/ReadVariableOp!^dense_28/StatefulPartitionedCall2^dense_28/kernel/Regularizer/Square/ReadVariableOp#^dropout_27/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2f
1dense_27/kernel/Regularizer/Square/ReadVariableOp1dense_27/kernel/Regularizer/Square/ReadVariableOp2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2f
1dense_28/kernel/Regularizer/Square/ReadVariableOp1dense_28/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
├
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1310104

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
ў
e
,__inference_dropout_27_layer_call_fn_1310357

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
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_27_layer_call_and_return_conditional_losses_13079762
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
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
serving_default_input_1:0         KK<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:б┴
Б

h2ptjl
_output
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+Г&call_and_return_all_conditional_losses
Д__call__
Е_default_save_signature
	Жcall"П	
_tf_keras_modelї{"name": "CNN3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN3", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN3"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
╩x
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
trainable_variables
regularization_losses
	variables
	keras_api
+З&call_and_return_all_conditional_losses
И__call__"нt
_tf_keras_sequentialОt{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_9_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 38, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_9_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_9_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25}, {"class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 26}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 32}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}, {"class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 37}]}}}
╫

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"░
_tf_keras_layerЦ{"name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
У
#iter

$beta_1

%beta_2
	&decay
'learning_ratemуmф(mх)mц*mч+mш,mщ-mъ.mы/mь0mэ1mю2mя3mЁ4mё5mЄvєvЇ(vї)vЎ*vў+v°,v∙-v·.v√/v№0v¤1v■2v 3vА4vБ5vВ"
	optimizer
Ц
(0
)1
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
412
513
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
ж
(0
)1
62
73
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
414
515
16
17"
trackable_list_wrapper
╬
8metrics
9layer_metrics
trainable_variables
:non_trainable_variables
regularization_losses
	variables

;layers
<layer_regularization_losses
Д__call__
Е_default_save_signature
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
-
Лserving_default"
signature_map
╪
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"╟
_tf_keras_layerн{"name": "lambda_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
─

Aaxis
	(gamma
)beta
6moving_mean
7moving_variance
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+О&call_and_return_all_conditional_losses
П__call__"ю
_tf_keras_layer╘{"name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
в

*kernel
+bias
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"√	
_tf_keras_layerс	{"name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
│
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
╓


,kernel
-bias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"п	
_tf_keras_layerХ	{"name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
│
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 47}}
╪


.kernel
/bias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"▒	
_tf_keras_layerЧ	{"name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
│
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 49}}
╓


0kernel
1bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"п	
_tf_keras_layerХ	{"name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 9, 9, 256]}}
│
btrainable_variables
cregularization_losses
d	variables
e	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 51}}
Б
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
+а&call_and_return_all_conditional_losses
б__call__"Ё
_tf_keras_layer╓{"name": "dropout_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 26}
Ш
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
+в&call_and_return_all_conditional_losses
г__call__"З
_tf_keras_layerэ{"name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 52}}
и	

2kernel
3bias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
+д&call_and_return_all_conditional_losses
е__call__"Б
_tf_keras_layerч{"name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 8192]}}
Б
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
+ж&call_and_return_all_conditional_losses
з__call__"Ё
_tf_keras_layer╓{"name": "dropout_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 32}
ж	

4kernel
5bias
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
+и&call_and_return_all_conditional_losses
й__call__" 
_tf_keras_layerх{"name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Б
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
+к&call_and_return_all_conditional_losses
л__call__"Ё
_tf_keras_layer╓{"name": "dropout_29", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 37}
Ж
(0
)1
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
412
513"
trackable_list_wrapper
8
м0
н1
о2"
trackable_list_wrapper
Ц
(0
)1
62
73
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
414
515"
trackable_list_wrapper
│
~metrics
layer_metrics
trainable_variables
Аnon_trainable_variables
regularization_losses
	variables
Бlayers
 Вlayer_regularization_losses
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_29/kernel
:2dense_29/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╡
Гmetrics
Дlayer_metrics
trainable_variables
Еnon_trainable_variables
 regularization_losses
!	variables
Жlayers
 Зlayer_regularization_losses
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_9/gamma
(:&2batch_normalization_9/beta
*:( 2conv2d_36/kernel
: 2conv2d_36/bias
+:) А2conv2d_37/kernel
:А2conv2d_37/bias
,:*АА2conv2d_38/kernel
:А2conv2d_38/bias
,:*АА2conv2d_39/kernel
:А2conv2d_39/bias
#:!
А@А2dense_27/kernel
:А2dense_27/bias
#:!
АА2dense_28/kernel
:А2dense_28/bias
1:/ (2!batch_normalization_9/moving_mean
5:3 (2%batch_normalization_9/moving_variance
0
И0
Й1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Кmetrics
Лlayer_metrics
=trainable_variables
Мnon_trainable_variables
>regularization_losses
?	variables
Нlayers
 Оlayer_regularization_losses
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
(0
)1
62
73"
trackable_list_wrapper
╡
Пmetrics
Рlayer_metrics
Btrainable_variables
Сnon_trainable_variables
Cregularization_losses
D	variables
Тlayers
 Уlayer_regularization_losses
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
(
м0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
╡
Фmetrics
Хlayer_metrics
Ftrainable_variables
Цnon_trainable_variables
Gregularization_losses
H	variables
Чlayers
 Шlayer_regularization_losses
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Щmetrics
Ъlayer_metrics
Jtrainable_variables
Ыnon_trainable_variables
Kregularization_losses
L	variables
Ьlayers
 Эlayer_regularization_losses
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
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
╡
Юmetrics
Яlayer_metrics
Ntrainable_variables
аnon_trainable_variables
Oregularization_losses
P	variables
бlayers
 вlayer_regularization_losses
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
гmetrics
дlayer_metrics
Rtrainable_variables
еnon_trainable_variables
Sregularization_losses
T	variables
жlayers
 зlayer_regularization_losses
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
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
╡
иmetrics
йlayer_metrics
Vtrainable_variables
кnon_trainable_variables
Wregularization_losses
X	variables
лlayers
 мlayer_regularization_losses
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
нmetrics
оlayer_metrics
Ztrainable_variables
пnon_trainable_variables
[regularization_losses
\	variables
░layers
 ▒layer_regularization_losses
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
╡
▓metrics
│layer_metrics
^trainable_variables
┤non_trainable_variables
_regularization_losses
`	variables
╡layers
 ╢layer_regularization_losses
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╖metrics
╕layer_metrics
btrainable_variables
╣non_trainable_variables
cregularization_losses
d	variables
║layers
 ╗layer_regularization_losses
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╝metrics
╜layer_metrics
ftrainable_variables
╛non_trainable_variables
gregularization_losses
h	variables
┐layers
 └layer_regularization_losses
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
┴metrics
┬layer_metrics
jtrainable_variables
├non_trainable_variables
kregularization_losses
l	variables
─layers
 ┼layer_regularization_losses
г__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
(
н0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
╡
╞metrics
╟layer_metrics
ntrainable_variables
╚non_trainable_variables
oregularization_losses
p	variables
╔layers
 ╩layer_regularization_losses
е__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╦metrics
╠layer_metrics
rtrainable_variables
═non_trainable_variables
sregularization_losses
t	variables
╬layers
 ╧layer_regularization_losses
з__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
(
о0"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
╡
╨metrics
╤layer_metrics
vtrainable_variables
╥non_trainable_variables
wregularization_losses
x	variables
╙layers
 ╘layer_regularization_losses
й__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╒metrics
╓layer_metrics
ztrainable_variables
╫non_trainable_variables
{regularization_losses
|	variables
╪layers
 ┘layer_regularization_losses
л__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
Ц
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
15"
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
╪

┌total

█count
▄	variables
▌	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 55}
Ы

▐total

▀count
р
_fn_kwargs
с	variables
т	keras_api"╧
_tf_keras_metric┤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
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
60
71"
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
м0"
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
(
н0"
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
о0"
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
┌0
█1"
trackable_list_wrapper
.
▄	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
▐0
▀1"
trackable_list_wrapper
.
с	variables"
_generic_user_object
':%	А2Adam/dense_29/kernel/m
 :2Adam/dense_29/bias/m
.:,2"Adam/batch_normalization_9/gamma/m
-:+2!Adam/batch_normalization_9/beta/m
/:- 2Adam/conv2d_36/kernel/m
!: 2Adam/conv2d_36/bias/m
0:. А2Adam/conv2d_37/kernel/m
": А2Adam/conv2d_37/bias/m
1:/АА2Adam/conv2d_38/kernel/m
": А2Adam/conv2d_38/bias/m
1:/АА2Adam/conv2d_39/kernel/m
": А2Adam/conv2d_39/bias/m
(:&
А@А2Adam/dense_27/kernel/m
!:А2Adam/dense_27/bias/m
(:&
АА2Adam/dense_28/kernel/m
!:А2Adam/dense_28/bias/m
':%	А2Adam/dense_29/kernel/v
 :2Adam/dense_29/bias/v
.:,2"Adam/batch_normalization_9/gamma/v
-:+2!Adam/batch_normalization_9/beta/v
/:- 2Adam/conv2d_36/kernel/v
!: 2Adam/conv2d_36/bias/v
0:. А2Adam/conv2d_37/kernel/v
": А2Adam/conv2d_37/bias/v
1:/АА2Adam/conv2d_38/kernel/v
": А2Adam/conv2d_38/bias/v
1:/АА2Adam/conv2d_39/kernel/v
": А2Adam/conv2d_39/bias/v
(:&
А@А2Adam/dense_27/kernel/v
!:А2Adam/dense_27/bias/v
(:&
АА2Adam/dense_28/kernel/v
!:А2Adam/dense_28/bias/v
╞2├
A__inference_CNN3_layer_call_and_return_conditional_losses_1308996
A__inference_CNN3_layer_call_and_return_conditional_losses_1309115
A__inference_CNN3_layer_call_and_return_conditional_losses_1309213
A__inference_CNN3_layer_call_and_return_conditional_losses_1309332┤
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
┌2╫
&__inference_CNN3_layer_call_fn_1309373
&__inference_CNN3_layer_call_fn_1309414
&__inference_CNN3_layer_call_fn_1309455
&__inference_CNN3_layer_call_fn_1309496┤
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
"__inference__wrapped_model_1307459╛
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
input_1         KK
З2Д
__inference_call_1249516
__inference_call_1249596
__inference_call_1249676│
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
Є2я
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309605
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309717
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309808
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309920└
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
Ж2Г
.__inference_sequential_9_layer_call_fn_1309957
.__inference_sequential_9_layer_call_fn_1309994
.__inference_sequential_9_layer_call_fn_1310031
.__inference_sequential_9_layer_call_fn_1310068└
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
я2ь
E__inference_dense_29_layer_call_and_return_conditional_losses_1310079в
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
*__inference_dense_29_layer_call_fn_1310088в
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
%__inference_signature_wrapper_1308898input_1"Ф
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
╘2╤
E__inference_lambda_9_layer_call_and_return_conditional_losses_1310096
E__inference_lambda_9_layer_call_and_return_conditional_losses_1310104└
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
Ю2Ы
*__inference_lambda_9_layer_call_fn_1310109
*__inference_lambda_9_layer_call_fn_1310114└
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
К2З
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310132
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310150
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310168
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310186┤
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
Ю2Ы
7__inference_batch_normalization_9_layer_call_fn_1310199
7__inference_batch_normalization_9_layer_call_fn_1310212
7__inference_batch_normalization_9_layer_call_fn_1310225
7__inference_batch_normalization_9_layer_call_fn_1310238┤
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
Ё2э
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1310261в
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
╒2╥
+__inference_conv2d_36_layer_call_fn_1310270в
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
╡2▓
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1307591р
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
Ъ2Ч
2__inference_max_pooling2d_36_layer_call_fn_1307597р
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
Ё2э
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1310281в
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
╒2╥
+__inference_conv2d_37_layer_call_fn_1310290в
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
╡2▓
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1307603р
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
Ъ2Ч
2__inference_max_pooling2d_37_layer_call_fn_1307609р
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
Ё2э
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1310301в
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
╒2╥
+__inference_conv2d_38_layer_call_fn_1310310в
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
╡2▓
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1307615р
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
Ъ2Ч
2__inference_max_pooling2d_38_layer_call_fn_1307621р
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
Ё2э
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1310321в
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
╒2╥
+__inference_conv2d_39_layer_call_fn_1310330в
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
╡2▓
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1307627р
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
Ъ2Ч
2__inference_max_pooling2d_39_layer_call_fn_1307633р
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
╠2╔
G__inference_dropout_27_layer_call_and_return_conditional_losses_1310335
G__inference_dropout_27_layer_call_and_return_conditional_losses_1310347┤
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
Ц2У
,__inference_dropout_27_layer_call_fn_1310352
,__inference_dropout_27_layer_call_fn_1310357┤
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
Ё2э
F__inference_flatten_9_layer_call_and_return_conditional_losses_1310363в
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
╒2╥
+__inference_flatten_9_layer_call_fn_1310368в
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
E__inference_dense_27_layer_call_and_return_conditional_losses_1310391в
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
*__inference_dense_27_layer_call_fn_1310400в
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
╠2╔
G__inference_dropout_28_layer_call_and_return_conditional_losses_1310405
G__inference_dropout_28_layer_call_and_return_conditional_losses_1310417┤
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
Ц2У
,__inference_dropout_28_layer_call_fn_1310422
,__inference_dropout_28_layer_call_fn_1310427┤
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
я2ь
E__inference_dense_28_layer_call_and_return_conditional_losses_1310450в
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
*__inference_dense_28_layer_call_fn_1310459в
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
╠2╔
G__inference_dropout_29_layer_call_and_return_conditional_losses_1310464
G__inference_dropout_29_layer_call_and_return_conditional_losses_1310476┤
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
Ц2У
,__inference_dropout_29_layer_call_fn_1310481
,__inference_dropout_29_layer_call_fn_1310486┤
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
__inference_loss_fn_0_1310497П
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
__inference_loss_fn_1_1310508П
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
__inference_loss_fn_2_1310519П
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
annotationsк *в ╜
A__inference_CNN3_layer_call_and_return_conditional_losses_1308996x()67*+,-./012345;в8
1в.
(К%
inputs         KK
p 
к "%в"
К
0         
Ъ ╜
A__inference_CNN3_layer_call_and_return_conditional_losses_1309115x()67*+,-./012345;в8
1в.
(К%
inputs         KK
p
к "%в"
К
0         
Ъ ╛
A__inference_CNN3_layer_call_and_return_conditional_losses_1309213y()67*+,-./012345<в9
2в/
)К&
input_1         KK
p 
к "%в"
К
0         
Ъ ╛
A__inference_CNN3_layer_call_and_return_conditional_losses_1309332y()67*+,-./012345<в9
2в/
)К&
input_1         KK
p
к "%в"
К
0         
Ъ Ц
&__inference_CNN3_layer_call_fn_1309373l()67*+,-./012345<в9
2в/
)К&
input_1         KK
p 
к "К         Х
&__inference_CNN3_layer_call_fn_1309414k()67*+,-./012345;в8
1в.
(К%
inputs         KK
p 
к "К         Х
&__inference_CNN3_layer_call_fn_1309455k()67*+,-./012345;в8
1в.
(К%
inputs         KK
p
к "К         Ц
&__inference_CNN3_layer_call_fn_1309496l()67*+,-./012345<в9
2в/
)К&
input_1         KK
p
к "К         к
"__inference__wrapped_model_1307459Г()67*+,-./0123458в5
.в+
)К&
input_1         KK
к "3к0
.
output_1"К
output_1         э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310132Ц()67MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310150Ц()67MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╚
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310168r()67;в8
1в.
(К%
inputs         KK
p 
к "-в*
#К 
0         KK
Ъ ╚
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1310186r()67;в8
1в.
(К%
inputs         KK
p
к "-в*
#К 
0         KK
Ъ ┼
7__inference_batch_normalization_9_layer_call_fn_1310199Й()67MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ┼
7__inference_batch_normalization_9_layer_call_fn_1310212Й()67MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           а
7__inference_batch_normalization_9_layer_call_fn_1310225e()67;в8
1в.
(К%
inputs         KK
p 
к " К         KKа
7__inference_batch_normalization_9_layer_call_fn_1310238e()67;в8
1в.
(К%
inputs         KK
p
к " К         KKw
__inference_call_1249516[()67*+,-./0123453в0
)в&
 К
inputsАKK
p
к "К	Аw
__inference_call_1249596[()67*+,-./0123453в0
)в&
 К
inputsАKK
p 
к "К	АЗ
__inference_call_1249676k()67*+,-./012345;в8
1в.
(К%
inputs         KK
p 
к "К         ╢
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1310261l*+7в4
-в*
(К%
inputs         KK
к "-в*
#К 
0         KK 
Ъ О
+__inference_conv2d_36_layer_call_fn_1310270_*+7в4
-в*
(К%
inputs         KK
к " К         KK ╖
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1310281m,-7в4
-в*
(К%
inputs         %% 
к ".в+
$К!
0         %%А
Ъ П
+__inference_conv2d_37_layer_call_fn_1310290`,-7в4
-в*
(К%
inputs         %% 
к "!К         %%А╕
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1310301n./8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Р
+__inference_conv2d_38_layer_call_fn_1310310a./8в5
.в+
)К&
inputs         А
к "!К         А╕
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1310321n018в5
.в+
)К&
inputs         		А
к ".в+
$К!
0         		А
Ъ Р
+__inference_conv2d_39_layer_call_fn_1310330a018в5
.в+
)К&
inputs         		А
к "!К         		Аз
E__inference_dense_27_layer_call_and_return_conditional_losses_1310391^230в-
&в#
!К
inputs         А@
к "&в#
К
0         А
Ъ 
*__inference_dense_27_layer_call_fn_1310400Q230в-
&в#
!К
inputs         А@
к "К         Аз
E__inference_dense_28_layer_call_and_return_conditional_losses_1310450^450в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ 
*__inference_dense_28_layer_call_fn_1310459Q450в-
&в#
!К
inputs         А
к "К         Аж
E__inference_dense_29_layer_call_and_return_conditional_losses_1310079]0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ ~
*__inference_dense_29_layer_call_fn_1310088P0в-
&в#
!К
inputs         А
к "К         ╣
G__inference_dropout_27_layer_call_and_return_conditional_losses_1310335n<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ╣
G__inference_dropout_27_layer_call_and_return_conditional_losses_1310347n<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ С
,__inference_dropout_27_layer_call_fn_1310352a<в9
2в/
)К&
inputs         А
p 
к "!К         АС
,__inference_dropout_27_layer_call_fn_1310357a<в9
2в/
)К&
inputs         А
p
к "!К         Ай
G__inference_dropout_28_layer_call_and_return_conditional_losses_1310405^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ й
G__inference_dropout_28_layer_call_and_return_conditional_losses_1310417^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Б
,__inference_dropout_28_layer_call_fn_1310422Q4в1
*в'
!К
inputs         А
p 
к "К         АБ
,__inference_dropout_28_layer_call_fn_1310427Q4в1
*в'
!К
inputs         А
p
к "К         Ай
G__inference_dropout_29_layer_call_and_return_conditional_losses_1310464^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ й
G__inference_dropout_29_layer_call_and_return_conditional_losses_1310476^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Б
,__inference_dropout_29_layer_call_fn_1310481Q4в1
*в'
!К
inputs         А
p 
к "К         АБ
,__inference_dropout_29_layer_call_fn_1310486Q4в1
*в'
!К
inputs         А
p
к "К         Ам
F__inference_flatten_9_layer_call_and_return_conditional_losses_1310363b8в5
.в+
)К&
inputs         А
к "&в#
К
0         А@
Ъ Д
+__inference_flatten_9_layer_call_fn_1310368U8в5
.в+
)К&
inputs         А
к "К         А@╣
E__inference_lambda_9_layer_call_and_return_conditional_losses_1310096p?в<
5в2
(К%
inputs         KK

 
p 
к "-в*
#К 
0         KK
Ъ ╣
E__inference_lambda_9_layer_call_and_return_conditional_losses_1310104p?в<
5в2
(К%
inputs         KK

 
p
к "-в*
#К 
0         KK
Ъ С
*__inference_lambda_9_layer_call_fn_1310109c?в<
5в2
(К%
inputs         KK

 
p 
к " К         KKС
*__inference_lambda_9_layer_call_fn_1310114c?в<
5в2
(К%
inputs         KK

 
p
к " К         KK<
__inference_loss_fn_0_1310497*в

в 
к "К <
__inference_loss_fn_1_13105082в

в 
к "К <
__inference_loss_fn_2_13105194в

в 
к "К Ё
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1307591ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_36_layer_call_fn_1307597СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1307603ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_37_layer_call_fn_1307609СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1307615ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_38_layer_call_fn_1307621СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1307627ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_39_layer_call_fn_1307633СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╚
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309605{()67*+,-./012345?в<
5в2
(К%
inputs         KK
p 

 
к "&в#
К
0         А
Ъ ╚
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309717{()67*+,-./012345?в<
5в2
(К%
inputs         KK
p

 
к "&в#
К
0         А
Ъ ╤
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309808Г()67*+,-./012345GвD
=в:
0К-
lambda_9_input         KK
p 

 
к "&в#
К
0         А
Ъ ╤
I__inference_sequential_9_layer_call_and_return_conditional_losses_1309920Г()67*+,-./012345GвD
=в:
0К-
lambda_9_input         KK
p

 
к "&в#
К
0         А
Ъ и
.__inference_sequential_9_layer_call_fn_1309957v()67*+,-./012345GвD
=в:
0К-
lambda_9_input         KK
p 

 
к "К         Аа
.__inference_sequential_9_layer_call_fn_1309994n()67*+,-./012345?в<
5в2
(К%
inputs         KK
p 

 
к "К         Аа
.__inference_sequential_9_layer_call_fn_1310031n()67*+,-./012345?в<
5в2
(К%
inputs         KK
p

 
к "К         Аи
.__inference_sequential_9_layer_call_fn_1310068v()67*+,-./012345GвD
=в:
0К-
lambda_9_input         KK
p

 
к "К         А╕
%__inference_signature_wrapper_1308898О()67*+,-./012345Cв@
в 
9к6
4
input_1)К&
input_1         KK"3к0
.
output_1"К
output_1         