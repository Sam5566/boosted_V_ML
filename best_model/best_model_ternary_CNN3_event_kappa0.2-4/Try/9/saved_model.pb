÷э
ді
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
ъ
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
epsilonfloat%Ј—8"&
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
Њ
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
ц
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
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718©Ў
{
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_19/kernel
t
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes
:	А*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
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
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А* 
shared_namedense_18/kernel
u
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel* 
_output_shapes
:
А@А*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
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
Ґ
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
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_19/kernel/m
В
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes
:	А*
dtype0
А
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
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
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_18/kernel/m
Г
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_18/bias/m
z
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_19/kernel/v
В
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes
:	А*
dtype0
А
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
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
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_18/kernel/v
Г
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_18/bias/v
z
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
Ў`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*У`
valueЙ`BЖ` B€_
К

h2ptjl
_output
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
®
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
Ў
!iter

"beta_1

#beta_2
	$decay
%learning_ratemЌmќ&mѕ'm–(m—)m“*m”+m‘,m’-m÷.m„/mЎ0mў1mЏvџv№&vЁ'vё(vя)vа*vб+vв,vг-vд.vе/vж0vз1vи
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
≠
trainable_variables
4non_trainable_variables
	variables
regularization_losses
5layer_regularization_losses
6layer_metrics

7layers
8metrics
 
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
Ч
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
h

.kernel
/bias
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
R
^trainable_variables
_	variables
`regularization_losses
a	keras_api
R
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
≠
trainable_variables
rnon_trainable_variables
	variables
regularization_losses
slayer_regularization_losses
tlayer_metrics

ulayers
vmetrics
NL
VARIABLE_VALUEdense_19/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_19/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
trainable_variables
wnon_trainable_variables
	variables
regularization_losses
xlayer_regularization_losses
ylayer_metrics

zlayers
{metrics
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
VARIABLE_VALUEdense_18/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_18/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
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
∞
9trainable_variables
~non_trainable_variables
:	variables
;regularization_losses
layer_regularization_losses
Аlayer_metrics
Бlayers
Вmetrics
 

&0
'1

&0
'1
22
33
 
≤
>trainable_variables
Гnon_trainable_variables
?	variables
@regularization_losses
 Дlayer_regularization_losses
Еlayer_metrics
Жlayers
Зmetrics

(0
)1

(0
)1
 
≤
Btrainable_variables
Иnon_trainable_variables
C	variables
Dregularization_losses
 Йlayer_regularization_losses
Кlayer_metrics
Лlayers
Мmetrics
 
 
 
≤
Ftrainable_variables
Нnon_trainable_variables
G	variables
Hregularization_losses
 Оlayer_regularization_losses
Пlayer_metrics
Рlayers
Сmetrics

*0
+1

*0
+1
 
≤
Jtrainable_variables
Тnon_trainable_variables
K	variables
Lregularization_losses
 Уlayer_regularization_losses
Фlayer_metrics
Хlayers
Цmetrics
 
 
 
≤
Ntrainable_variables
Чnon_trainable_variables
O	variables
Pregularization_losses
 Шlayer_regularization_losses
Щlayer_metrics
Ъlayers
Ыmetrics

,0
-1

,0
-1
 
≤
Rtrainable_variables
Ьnon_trainable_variables
S	variables
Tregularization_losses
 Эlayer_regularization_losses
Юlayer_metrics
Яlayers
†metrics
 
 
 
≤
Vtrainable_variables
°non_trainable_variables
W	variables
Xregularization_losses
 Ґlayer_regularization_losses
£layer_metrics
§layers
•metrics

.0
/1

.0
/1
 
≤
Ztrainable_variables
¶non_trainable_variables
[	variables
\regularization_losses
 Іlayer_regularization_losses
®layer_metrics
©layers
™metrics
 
 
 
≤
^trainable_variables
Ђnon_trainable_variables
_	variables
`regularization_losses
 ђlayer_regularization_losses
≠layer_metrics
Ѓlayers
ѓmetrics
 
 
 
≤
btrainable_variables
∞non_trainable_variables
c	variables
dregularization_losses
 ±layer_regularization_losses
≤layer_metrics
≥layers
іmetrics
 
 
 
≤
ftrainable_variables
µnon_trainable_variables
g	variables
hregularization_losses
 ґlayer_regularization_losses
Јlayer_metrics
Єlayers
єmetrics

00
11

00
11
 
≤
jtrainable_variables
Їnon_trainable_variables
k	variables
lregularization_losses
 їlayer_regularization_losses
Љlayer_metrics
љlayers
Њmetrics
 
 
 
≤
ntrainable_variables
њnon_trainable_variables
o	variables
pregularization_losses
 јlayer_regularization_losses
Ѕlayer_metrics
¬layers
√metrics
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

ƒtotal

≈count
∆	variables
«	keras_api
I

»total

…count
 
_fn_kwargs
Ћ	variables
ћ	keras_api
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
ƒ0
≈1

∆	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

»0
…1

Ћ	variables
qo
VARIABLE_VALUEAdam/dense_19/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_19/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_18/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_18/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_19/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_19/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_18/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_18/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:€€€€€€€€€KK*
dtype0*$
shape:€€€€€€€€€KK
£
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_1060002
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
р
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp+Adam/conv2d_36/kernel/m/Read/ReadVariableOp)Adam/conv2d_36/bias/m/Read/ReadVariableOp+Adam/conv2d_37/kernel/m/Read/ReadVariableOp)Adam/conv2d_37/bias/m/Read/ReadVariableOp+Adam/conv2d_38/kernel/m/Read/ReadVariableOp)Adam/conv2d_38/bias/m/Read/ReadVariableOp+Adam/conv2d_39/kernel/m/Read/ReadVariableOp)Adam/conv2d_39/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp+Adam/conv2d_36/kernel/v/Read/ReadVariableOp)Adam/conv2d_36/bias/v/Read/ReadVariableOp+Adam/conv2d_37/kernel/v/Read/ReadVariableOp)Adam/conv2d_37/bias/v/Read/ReadVariableOp+Adam/conv2d_38/kernel/v/Read/ReadVariableOp)Adam/conv2d_38/bias/v/Read/ReadVariableOp+Adam/conv2d_39/kernel/v/Read/ReadVariableOp)Adam/conv2d_39/bias/v/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOpConst*B
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
GPU2 *0J 8В *)
f$R"
 __inference__traced_save_1061557
«
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_19/kerneldense_19/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_9/gammabatch_normalization_9/betaconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasdense_18/kerneldense_18/bias!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancetotalcounttotal_1count_1Adam/dense_19/kernel/mAdam/dense_19/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/conv2d_36/kernel/mAdam/conv2d_36/bias/mAdam/conv2d_37/kernel/mAdam/conv2d_37/bias/mAdam/conv2d_38/kernel/mAdam/conv2d_38/bias/mAdam/conv2d_39/kernel/mAdam/conv2d_39/bias/mAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/vAdam/dense_19/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/conv2d_36/kernel/vAdam/conv2d_36/bias/vAdam/conv2d_37/kernel/vAdam/conv2d_37/bias/vAdam/conv2d_38/kernel/vAdam/conv2d_38/bias/vAdam/conv2d_39/kernel/vAdam/conv2d_39/bias/vAdam/dense_18/kernel/vAdam/dense_18/bias/v*A
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
GPU2 *0J 8В *,
f'R%
#__inference__traced_restore_1061726ле
§
ґ
&__inference_CNN3_layer_call_fn_1060440

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

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCall±
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
:€€€€€€€€€*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_10596392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
™(
П
A__inference_CNN3_layer_call_and_return_conditional_losses_1059639

inputs"
sequential_9_1059580:"
sequential_9_1059582:"
sequential_9_1059584:"
sequential_9_1059586:.
sequential_9_1059588: "
sequential_9_1059590: /
sequential_9_1059592: А#
sequential_9_1059594:	А0
sequential_9_1059596:АА#
sequential_9_1059598:	А0
sequential_9_1059600:АА#
sequential_9_1059602:	А(
sequential_9_1059604:
А@А#
sequential_9_1059606:	А#
dense_19_1059621:	А
dense_19_1059623:
identityИҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpҐ dense_19/StatefulPartitionedCallҐ$sequential_9/StatefulPartitionedCall—
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_1059580sequential_9_1059582sequential_9_1059584sequential_9_1059586sequential_9_1059588sequential_9_1059590sequential_9_1059592sequential_9_1059594sequential_9_1059596sequential_9_1059598sequential_9_1059600sequential_9_1059602sequential_9_1059604sequential_9_1059606*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_10591042&
$sequential_9/StatefulPartitionedCall√
 dense_19/StatefulPartitionedCallStatefulPartitionedCall-sequential_9/StatefulPartitionedCall:output:0dense_19_1059621dense_19_1059623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_19_layer_call_and_return_conditional_losses_10596202"
 dense_19/StatefulPartitionedCall≈
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1059588*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulљ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1059604* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul∞
IdentityIdentity)dense_19/StatefulPartitionedCall:output:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
Ю
Б
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1059003

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
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
:€€€€€€€€€%%А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€%%А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€%% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€%% 
 
_user_specified_nameinputs
–
Ґ
+__inference_conv2d_37_layer_call_fn_1061216

inputs"
unknown: А
	unknown_0:	А
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€%%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_10590032
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€%%А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€%% : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€%% 
 
_user_specified_nameinputs
а
N
2__inference_max_pooling2d_38_layer_call_fn_1058912

inputs
identityу
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_10589062
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ґ
f
G__inference_dropout_19_layer_call_and_return_conditional_losses_1061343

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
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
к
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1059059

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
є

ч
E__inference_dense_19_layer_call_and_return_conditional_losses_1061005

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ґ
В
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1059021

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
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
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
г
F
*__inference_lambda_9_layer_call_fn_1061035

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_10589392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€KK:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ј
Ѕ
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061076

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ю
Б
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1061207

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
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
:€€€€€€€€€%%А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€%%А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€%% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€%% 
 
_user_specified_nameinputs
Ш
e
G__inference_dropout_18_layer_call_and_return_conditional_losses_1061261

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
•u
€
__inference_call_1021991

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
4sequential_9_dense_18_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_18_biasadd_readvariableop_resource:	А:
'dense_19_matmul_readvariableop_resource:	А6
(dense_19_biasadd_readvariableop_resource:
identityИҐdense_19/BiasAdd/ReadVariableOpҐdense_19/MatMul/ReadVariableOpҐBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_9/batch_normalization_9/ReadVariableOpҐ3sequential_9/batch_normalization_9/ReadVariableOp_1Ґ-sequential_9/conv2d_36/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_36/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_37/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_37/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_38/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_38/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_39/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_39/Conv2D/ReadVariableOpҐ,sequential_9/dense_18/BiasAdd/ReadVariableOpҐ+sequential_9/dense_18/MatMul/ReadVariableOpѓ
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack≥
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1≥
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2г
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_sliceЁ
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpг
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
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ї
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3Џ
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
sequential_9/conv2d_36/Conv2D—
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp№
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2 
sequential_9/conv2d_36/BiasAddЭ
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_9/conv2d_36/Reluй
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPoolџ
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
sequential_9/conv2d_37/Conv2D“
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpЁ
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2 
sequential_9/conv2d_37/BiasAddЮ
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_9/conv2d_37/Reluк
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool№
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
sequential_9/conv2d_38/Conv2D“
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpЁ
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_9/conv2d_38/BiasAddЮ
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_9/conv2d_38/Reluк
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool№
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
sequential_9/conv2d_39/Conv2D“
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpЁ
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А2 
sequential_9/conv2d_39/BiasAddЮ
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*(
_output_shapes
:А		А2
sequential_9/conv2d_39/Reluк
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool≥
 sequential_9/dropout_18/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*(
_output_shapes
:АА2"
 sequential_9/dropout_18/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
sequential_9/flatten_9/Const»
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_18/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0* 
_output_shapes
:
АА@2 
sequential_9/flatten_9/Reshape—
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOpѕ
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_18/MatMulѕ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOp“
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_18/BiasAddУ
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_18/Relu•
 sequential_9/dropout_19/IdentityIdentity(sequential_9/dense_18/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_9/dropout_19/Identity©
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp©
dense_19/MatMulMatMul)sequential_9/dropout_19/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_19/MatMulІ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOpЭ
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_19/BiasAddt
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_19/Softmaxш
IdentityIdentitydense_19/Softmax:softmax:0 ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2И
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
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
а
N
2__inference_max_pooling2d_39_layer_call_fn_1058924

inputs
identityу
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_10589182
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ґ
В
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1059039

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
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
:€€€€€€€€€		А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€		А
 
_user_specified_nameinputs
З
ґ
%__inference_signature_wrapper_1060002
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

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallУ
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
:€€€€€€€€€*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_10587502
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€KK
!
_user_specified_name	input_1
√
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1061022

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
strided_slice/stack_2э
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€KK:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
∆m
ю
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060771
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
'dense_18_matmul_readvariableop_resource:
А@А7
(dense_18_biasadd_readvariableop_resource:	А
identityИҐ5batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_9/ReadVariableOpҐ&batch_normalization_9/ReadVariableOp_1Ґ conv2d_36/BiasAdd/ReadVariableOpҐconv2d_36/Conv2D/ReadVariableOpҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ conv2d_37/BiasAdd/ReadVariableOpҐconv2d_37/Conv2D/ReadVariableOpҐ conv2d_38/BiasAdd/ReadVariableOpҐconv2d_38/Conv2D/ReadVariableOpҐ conv2d_39/BiasAdd/ReadVariableOpҐconv2d_39/Conv2D/ReadVariableOpҐdense_18/BiasAdd/ReadVariableOpҐdense_18/MatMul/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpХ
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
lambda_9/strided_slice/stack_2≤
lambda_9/strided_sliceStridedSlicelambda_9_input%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2
lambda_9/strided_sliceґ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOpЉ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1й
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1з
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3≥
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOpе
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
conv2d_36/Conv2D™
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp∞
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
conv2d_36/Relu 
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPoolі
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_37/Conv2D/ReadVariableOpЁ
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
conv2d_37/Conv2DЂ
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp±
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
conv2d_37/ReluЋ
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPoolµ
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_38/Conv2D/ReadVariableOpЁ
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_38/Conv2DЂ
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp±
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_38/ReluЋ
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolµ
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_39/Conv2D/ReadVariableOpЁ
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
conv2d_39/Conv2DЂ
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp±
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
conv2d_39/ReluЋ
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPoolФ
dropout_18/IdentityIdentity!max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_18/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
flatten_9/ConstЬ
flatten_9/ReshapeReshapedropout_18/Identity:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2
flatten_9/Reshape™
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_18/MatMul/ReadVariableOp£
dense_18/MatMulMatMulflatten_9/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/MatMul®
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_18/BiasAdd/ReadVariableOp¶
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/ReluЖ
dropout_19/IdentityIdentitydense_18/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_19/Identityў
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul–
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulу
IdentityIdentitydropout_19/Identity:output:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 2n
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
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:€€€€€€€€€KK
(
_user_specified_namelambda_9_input
ч
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_1059194

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/y«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
√
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1061030

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
strided_slice/stack_2э
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€KK:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
у
Н
.__inference_sequential_9_layer_call_fn_1060994
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

unknown_12:	А
identityИҐStatefulPartitionedCall§
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
:€€€€€€€€€А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_10593932
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€KK
(
_user_specified_namelambda_9_input
М
Э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1058772

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
”
£
+__inference_conv2d_38_layer_call_fn_1061236

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_10590212
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
”
£
+__inference_conv2d_39_layer_call_fn_1061256

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_10590392
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€		А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€		А
 
_user_specified_nameinputs
шR
п
I__inference_sequential_9_layer_call_and_return_conditional_losses_1059104

inputs+
batch_normalization_9_1058959:+
batch_normalization_9_1058961:+
batch_normalization_9_1058963:+
batch_normalization_9_1058965:+
conv2d_36_1058986: 
conv2d_36_1058988: ,
conv2d_37_1059004: А 
conv2d_37_1059006:	А-
conv2d_38_1059022:АА 
conv2d_38_1059024:	А-
conv2d_39_1059040:АА 
conv2d_39_1059042:	А$
dense_18_1059079:
А@А
dense_18_1059081:	А
identityИҐ-batch_normalization_9/StatefulPartitionedCallҐ!conv2d_36/StatefulPartitionedCallҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ!conv2d_37/StatefulPartitionedCallҐ!conv2d_38/StatefulPartitionedCallҐ!conv2d_39/StatefulPartitionedCallҐ dense_18/StatefulPartitionedCallҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpв
lambda_9/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_10589392
lambda_9/PartitionedCall¬
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall!lambda_9/PartitionedCall:output:0batch_normalization_9_1058959batch_normalization_9_1058961batch_normalization_9_1058963batch_normalization_9_1058965*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_10589582/
-batch_normalization_9/StatefulPartitionedCallў
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_36_1058986conv2d_36_1058988*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_10589852#
!conv2d_36/StatefulPartitionedCallЮ
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€%% * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_10588822"
 max_pooling2d_36/PartitionedCallЌ
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_37_1059004conv2d_37_1059006*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€%%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_10590032#
!conv2d_37/StatefulPartitionedCallЯ
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_10588942"
 max_pooling2d_37/PartitionedCallЌ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_38_1059022conv2d_38_1059024*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_10590212#
!conv2d_38/StatefulPartitionedCallЯ
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_10589062"
 max_pooling2d_38/PartitionedCallЌ
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_1059040conv2d_39_1059042*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_10590392#
!conv2d_39/StatefulPartitionedCallЯ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_10589182"
 max_pooling2d_39/PartitionedCallМ
dropout_18/PartitionedCallPartitionedCall)max_pooling2d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_10590512
dropout_18/PartitionedCallы
flatten_9/PartitionedCallPartitionedCall#dropout_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_10590592
flatten_9/PartitionedCallє
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_18_1059079dense_18_1059081*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_18_layer_call_and_return_conditional_losses_10590782"
 dense_18/StatefulPartitionedCallД
dropout_19/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_10590892
dropout_19/PartitionedCall¬
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_36_1058986*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulє
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_1059079* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulƒ
IdentityIdentity#dropout_19/PartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ш
Ѕ
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061112

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ў
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ю
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€KK: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
а
N
2__inference_max_pooling2d_36_layer_call_fn_1058888

inputs
identityу
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_10588822
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ё
Е
.__inference_sequential_9_layer_call_fn_1060928

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

unknown_12:	А
identityИҐStatefulPartitionedCallЮ
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
:€€€€€€€€€А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_10591042
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ч
е
"__inference__wrapped_model_1058750
input_1
cnn3_1058716:
cnn3_1058718:
cnn3_1058720:
cnn3_1058722:&
cnn3_1058724: 
cnn3_1058726: '
cnn3_1058728: А
cnn3_1058730:	А(
cnn3_1058732:АА
cnn3_1058734:	А(
cnn3_1058736:АА
cnn3_1058738:	А 
cnn3_1058740:
А@А
cnn3_1058742:	А
cnn3_1058744:	А
cnn3_1058746:
identityИҐCNN3/StatefulPartitionedCallј
CNN3/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn3_1058716cnn3_1058718cnn3_1058720cnn3_1058722cnn3_1058724cnn3_1058726cnn3_1058728cnn3_1058730cnn3_1058732cnn3_1058734cnn3_1058736cnn3_1058738cnn3_1058740cnn3_1058742cnn3_1058744cnn3_1058746*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *!
fR
__inference_call_10201202
CNN3/StatefulPartitionedCallШ
IdentityIdentity%CNN3/StatefulPartitionedCall:output:0^CNN3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2<
CNN3/StatefulPartitionedCallCNN3/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€KK
!
_user_specified_name	input_1
а
N
2__inference_max_pooling2d_37_layer_call_fn_1058900

inputs
identityу
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_10588942
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ш
e
G__inference_dropout_18_layer_call_and_return_conditional_losses_1059051

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ГV
є
I__inference_sequential_9_layer_call_and_return_conditional_losses_1059393

inputs+
batch_normalization_9_1059339:+
batch_normalization_9_1059341:+
batch_normalization_9_1059343:+
batch_normalization_9_1059345:+
conv2d_36_1059348: 
conv2d_36_1059350: ,
conv2d_37_1059354: А 
conv2d_37_1059356:	А-
conv2d_38_1059360:АА 
conv2d_38_1059362:	А-
conv2d_39_1059366:АА 
conv2d_39_1059368:	А$
dense_18_1059374:
А@А
dense_18_1059376:	А
identityИҐ-batch_normalization_9/StatefulPartitionedCallҐ!conv2d_36/StatefulPartitionedCallҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ!conv2d_37/StatefulPartitionedCallҐ!conv2d_38/StatefulPartitionedCallҐ!conv2d_39/StatefulPartitionedCallҐ dense_18/StatefulPartitionedCallҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpҐ"dropout_18/StatefulPartitionedCallҐ"dropout_19/StatefulPartitionedCallв
lambda_9/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_10592972
lambda_9/PartitionedCallј
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall!lambda_9/PartitionedCall:output:0batch_normalization_9_1059339batch_normalization_9_1059341batch_normalization_9_1059343batch_normalization_9_1059345*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_10592702/
-batch_normalization_9/StatefulPartitionedCallў
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_36_1059348conv2d_36_1059350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_10589852#
!conv2d_36/StatefulPartitionedCallЮ
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€%% * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_10588822"
 max_pooling2d_36/PartitionedCallЌ
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_37_1059354conv2d_37_1059356*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€%%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_10590032#
!conv2d_37/StatefulPartitionedCallЯ
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_10588942"
 max_pooling2d_37/PartitionedCallЌ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_38_1059360conv2d_38_1059362*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_10590212#
!conv2d_38/StatefulPartitionedCallЯ
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_10589062"
 max_pooling2d_38/PartitionedCallЌ
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_1059366conv2d_39_1059368*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_10590392#
!conv2d_39/StatefulPartitionedCallЯ
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_10589182"
 max_pooling2d_39/PartitionedCall§
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_10591942$
"dropout_18/StatefulPartitionedCallГ
flatten_9/PartitionedCallPartitionedCall+dropout_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_10590592
flatten_9/PartitionedCallє
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_18_1059374dense_18_1059376*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_18_layer_call_and_return_conditional_losses_10590782"
 dense_18/StatefulPartitionedCallЅ
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_10591552$
"dropout_19/StatefulPartitionedCall¬
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_36_1059348*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulє
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_18_1059374* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulЦ
IdentityIdentity+dropout_19/StatefulPartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall2^dense_18/kernel/Regularizer/Square/ReadVariableOp#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ш
e
G__inference_dropout_19_layer_call_and_return_conditional_losses_1061331

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ї
≠
E__inference_dense_18_layer_call_and_return_conditional_losses_1059078

inputs2
matmul_readvariableop_resource:
А@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relu«
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulћ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А@
 
_user_specified_nameinputs
ш
Ѕ
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1059270

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ў
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ю
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€KK: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ƒ
Э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1058958

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€KK: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
√
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1059297

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
strided_slice/stack_2э
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€KK:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
о
“
7__inference_batch_normalization_9_layer_call_fn_1061138

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_10588162
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ј
Ѕ
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1058816

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ђ≠
ы
A__inference_CNN3_layer_call_and_return_conditional_losses_1060184

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
4sequential_9_dense_18_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_18_biasadd_readvariableop_resource:	А:
'dense_19_matmul_readvariableop_resource:	А6
(dense_19_biasadd_readvariableop_resource:
identityИҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpҐdense_19/BiasAdd/ReadVariableOpҐdense_19/MatMul/ReadVariableOpҐ1sequential_9/batch_normalization_9/AssignNewValueҐ3sequential_9/batch_normalization_9/AssignNewValue_1ҐBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_9/batch_normalization_9/ReadVariableOpҐ3sequential_9/batch_normalization_9/ReadVariableOp_1Ґ-sequential_9/conv2d_36/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_36/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_37/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_37/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_38/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_38/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_39/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_39/Conv2D/ReadVariableOpҐ,sequential_9/dense_18/BiasAdd/ReadVariableOpҐ+sequential_9/dense_18/MatMul/ReadVariableOpѓ
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack≥
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1≥
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2л
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_sliceЁ
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpг
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
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1–
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<25
3sequential_9/batch_normalization_9/FusedBatchNormV3с
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_9/batch_normalization_9/AssignNewValueэ
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_9/batch_normalization_9/AssignNewValue_1Џ
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D—
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpд
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2 
sequential_9/conv2d_36/BiasAdd•
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
sequential_9/conv2d_36/Reluс
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPoolџ
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D“
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpе
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2 
sequential_9/conv2d_37/BiasAdd¶
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
sequential_9/conv2d_37/Reluт
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool№
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D“
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpе
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_9/conv2d_38/BiasAdd¶
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_9/conv2d_38/Reluт
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool№
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D“
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpе
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2 
sequential_9/conv2d_39/BiasAdd¶
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
sequential_9/conv2d_39/Reluт
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPoolУ
%sequential_9/dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2'
%sequential_9/dropout_18/dropout/Constм
#sequential_9/dropout_18/dropout/MulMul.sequential_9/max_pooling2d_39/MaxPool:output:0.sequential_9/dropout_18/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2%
#sequential_9/dropout_18/dropout/Mulђ
%sequential_9/dropout_18/dropout/ShapeShape.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_18/dropout/ShapeЕ
<sequential_9/dropout_18/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_18/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype02>
<sequential_9/dropout_18/dropout/random_uniform/RandomUniform•
.sequential_9/dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=20
.sequential_9/dropout_18/dropout/GreaterEqual/yІ
,sequential_9/dropout_18/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_18/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_18/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2.
,sequential_9/dropout_18/dropout/GreaterEqual–
$sequential_9/dropout_18/dropout/CastCast0sequential_9/dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2&
$sequential_9/dropout_18/dropout/Castг
%sequential_9/dropout_18/dropout/Mul_1Mul'sequential_9/dropout_18/dropout/Mul:z:0(sequential_9/dropout_18/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2'
%sequential_9/dropout_18/dropout/Mul_1Н
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
sequential_9/flatten_9/Const–
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_18/dropout/Mul_1:z:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2 
sequential_9/flatten_9/Reshape—
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOp„
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/MatMulѕ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOpЏ
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/BiasAddЫ
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/ReluУ
%sequential_9/dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_19/dropout/Constё
#sequential_9/dropout_19/dropout/MulMul(sequential_9/dense_18/Relu:activations:0.sequential_9/dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2%
#sequential_9/dropout_19/dropout/Mul¶
%sequential_9/dropout_19/dropout/ShapeShape(sequential_9/dense_18/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_19/dropout/Shapeэ
<sequential_9/dropout_19/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02>
<sequential_9/dropout_19/dropout/random_uniform/RandomUniform•
.sequential_9/dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_19/dropout/GreaterEqual/yЯ
,sequential_9/dropout_19/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_19/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2.
,sequential_9/dropout_19/dropout/GreaterEqual»
$sequential_9/dropout_19/dropout/CastCast0sequential_9/dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2&
$sequential_9/dropout_19/dropout/Castџ
%sequential_9/dropout_19/dropout/Mul_1Mul'sequential_9/dropout_19/dropout/Mul:z:0(sequential_9/dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%sequential_9/dropout_19/dropout/Mul_1©
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp±
dense_19/MatMulMatMul)sequential_9/dropout_19/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/MatMulІ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp•
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/BiasAdd|
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/Softmaxж
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulЁ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul”
IdentityIdentitydense_19/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2f
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
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
≠
i
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1058882

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ЫИ
∆
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060694

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
'dense_18_matmul_readvariableop_resource:
А@А7
(dense_18_biasadd_readvariableop_resource:	А
identityИҐ$batch_normalization_9/AssignNewValueҐ&batch_normalization_9/AssignNewValue_1Ґ5batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_9/ReadVariableOpҐ&batch_normalization_9/ReadVariableOp_1Ґ conv2d_36/BiasAdd/ReadVariableOpҐconv2d_36/Conv2D/ReadVariableOpҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ conv2d_37/BiasAdd/ReadVariableOpҐconv2d_37/Conv2D/ReadVariableOpҐ conv2d_38/BiasAdd/ReadVariableOpҐconv2d_38/Conv2D/ReadVariableOpҐ conv2d_39/BiasAdd/ReadVariableOpҐconv2d_39/Conv2D/ReadVariableOpҐdense_18/BiasAdd/ReadVariableOpҐdense_18/MatMul/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpХ
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
lambda_9/strided_slice/stack_2™
lambda_9/strided_sliceStridedSliceinputs%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2
lambda_9/strided_sliceґ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOpЉ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1й
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1х
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2(
&batch_normalization_9/FusedBatchNormV3∞
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValueЉ
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1≥
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOpе
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
conv2d_36/Conv2D™
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp∞
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
conv2d_36/Relu 
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPoolі
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_37/Conv2D/ReadVariableOpЁ
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
conv2d_37/Conv2DЂ
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp±
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
conv2d_37/ReluЋ
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPoolµ
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_38/Conv2D/ReadVariableOpЁ
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_38/Conv2DЂ
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp±
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_38/ReluЋ
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolµ
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_39/Conv2D/ReadVariableOpЁ
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
conv2d_39/Conv2DЂ
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp±
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
conv2d_39/ReluЋ
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPooly
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_18/dropout/ConstЄ
dropout_18/dropout/MulMul!max_pooling2d_39/MaxPool:output:0!dropout_18/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_18/dropout/MulЕ
dropout_18/dropout/ShapeShape!max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shapeё
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype021
/dropout_18/dropout/random_uniform/RandomUniformЛ
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_18/dropout/GreaterEqual/yу
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2!
dropout_18/dropout/GreaterEqual©
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout_18/dropout/Castѓ
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_18/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
flatten_9/ConstЬ
flatten_9/ReshapeReshapedropout_18/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2
flatten_9/Reshape™
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_18/MatMul/ReadVariableOp£
dense_18/MatMulMatMulflatten_9/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/MatMul®
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_18/BiasAdd/ReadVariableOp¶
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/Reluy
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_19/dropout/Const™
dropout_19/dropout/MulMuldense_18/Relu:activations:0!dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_19/dropout/Mul
dropout_19/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shape÷
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype021
/dropout_19/dropout/random_uniform/RandomUniformЛ
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_19/dropout/GreaterEqual/yл
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
dropout_19/dropout/GreaterEqual°
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_19/dropout/CastІ
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_19/dropout/Mul_1ў
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul–
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul√
IdentityIdentitydropout_19/dropout/Mul_1:z:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 2L
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
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
≠
i
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1058918

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
І
Ј
&__inference_CNN3_layer_call_fn_1060403
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

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCall≤
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
:€€€€€€€€€*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_10596392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€KK
!
_user_specified_name	input_1
ЩМ
Т
A__inference_CNN3_layer_call_and_return_conditional_losses_1060268
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
4sequential_9_dense_18_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_18_biasadd_readvariableop_resource:	А:
'dense_19_matmul_readvariableop_resource:	А6
(dense_19_biasadd_readvariableop_resource:
identityИҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpҐdense_19/BiasAdd/ReadVariableOpҐdense_19/MatMul/ReadVariableOpҐBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_9/batch_normalization_9/ReadVariableOpҐ3sequential_9/batch_normalization_9/ReadVariableOp_1Ґ-sequential_9/conv2d_36/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_36/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_37/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_37/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_38/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_38/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_39/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_39/Conv2D/ReadVariableOpҐ,sequential_9/dense_18/BiasAdd/ReadVariableOpҐ+sequential_9/dense_18/MatMul/ReadVariableOpѓ
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack≥
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1≥
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2м
#sequential_9/lambda_9/strided_sliceStridedSliceinput_12sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_sliceЁ
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpг
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
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¬
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3Џ
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D—
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpд
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2 
sequential_9/conv2d_36/BiasAdd•
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
sequential_9/conv2d_36/Reluс
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPoolџ
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D“
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpе
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2 
sequential_9/conv2d_37/BiasAdd¶
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
sequential_9/conv2d_37/Reluт
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool№
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D“
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpе
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_9/conv2d_38/BiasAdd¶
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_9/conv2d_38/Reluт
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool№
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D“
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpе
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2 
sequential_9/conv2d_39/BiasAdd¶
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
sequential_9/conv2d_39/Reluт
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPoolї
 sequential_9/dropout_18/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 sequential_9/dropout_18/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
sequential_9/flatten_9/Const–
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_18/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2 
sequential_9/flatten_9/Reshape—
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOp„
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/MatMulѕ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOpЏ
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/BiasAddЫ
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/Relu≠
 sequential_9/dropout_19/IdentityIdentity(sequential_9/dense_18/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 sequential_9/dropout_19/Identity©
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp±
dense_19/MatMulMatMul)sequential_9/dropout_19/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/MatMulІ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp•
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/BiasAdd|
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/Softmaxж
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulЁ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulй
IdentityIdentitydense_19/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2И
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
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€KK
!
_user_specified_name	input_1
Ы
Љ
__inference_loss_fn_0_1061364U
;conv2d_36_kernel_regularizer_square_readvariableop_resource: 
identityИҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpм
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_36_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
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
ƒ
Э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061094

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€KK: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ЦМ
С
A__inference_CNN3_layer_call_and_return_conditional_losses_1060086

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
4sequential_9_dense_18_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_18_biasadd_readvariableop_resource:	А:
'dense_19_matmul_readvariableop_resource:	А6
(dense_19_biasadd_readvariableop_resource:
identityИҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpҐdense_19/BiasAdd/ReadVariableOpҐdense_19/MatMul/ReadVariableOpҐBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_9/batch_normalization_9/ReadVariableOpҐ3sequential_9/batch_normalization_9/ReadVariableOp_1Ґ-sequential_9/conv2d_36/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_36/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_37/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_37/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_38/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_38/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_39/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_39/Conv2D/ReadVariableOpҐ,sequential_9/dense_18/BiasAdd/ReadVariableOpҐ+sequential_9/dense_18/MatMul/ReadVariableOpѓ
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack≥
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1≥
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2л
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_sliceЁ
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpг
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
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¬
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3Џ
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D—
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpд
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2 
sequential_9/conv2d_36/BiasAdd•
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
sequential_9/conv2d_36/Reluс
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPoolџ
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D“
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpе
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2 
sequential_9/conv2d_37/BiasAdd¶
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
sequential_9/conv2d_37/Reluт
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool№
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D“
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpе
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_9/conv2d_38/BiasAdd¶
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_9/conv2d_38/Reluт
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool№
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D“
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpе
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2 
sequential_9/conv2d_39/BiasAdd¶
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
sequential_9/conv2d_39/Reluт
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPoolї
 sequential_9/dropout_18/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 sequential_9/dropout_18/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
sequential_9/flatten_9/Const–
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_18/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2 
sequential_9/flatten_9/Reshape—
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOp„
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/MatMulѕ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOpЏ
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/BiasAddЫ
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/Relu≠
 sequential_9/dropout_19/IdentityIdentity(sequential_9/dense_18/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 sequential_9/dropout_19/Identity©
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp±
dense_19/MatMulMatMul)sequential_9/dropout_19/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/MatMulІ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp•
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/BiasAdd|
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/Softmaxж
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulЁ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulй
IdentityIdentitydense_19/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2И
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
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ј
і
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1058985

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
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
:€€€€€€€€€KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
Reluѕ
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul‘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ћ
†
+__inference_conv2d_36_layer_call_fn_1061196

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_10589852
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€KK: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
®
“
7__inference_batch_normalization_9_layer_call_fn_1061151

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_10589582
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€KK: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
х
Н
.__inference_sequential_9_layer_call_fn_1060895
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

unknown_12:	А
identityИҐStatefulPartitionedCall¶
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
:€€€€€€€€€А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_10591042
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€KK
(
_user_specified_namelambda_9_input
Ґ
ґ
&__inference_CNN3_layer_call_fn_1060477

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

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallѓ
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
:€€€€€€€€€*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_10597732
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
й
і
__inference_loss_fn_1_1061375N
:dense_18_kernel_regularizer_square_readvariableop_resource:
А@А
identityИҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpг
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_18_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulЪ
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
Ґ
В
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1061227

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
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
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
р
“
7__inference_batch_normalization_9_layer_call_fn_1061125

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_10587722
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ
H
,__inference_dropout_19_layer_call_fn_1061348

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_10590892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
©k
ќ
 __inference__traced_save_1061557
file_prefix.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop(
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
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableopA
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
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableopA
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
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename»
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*Џ
value–BЌ6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesь
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop2savev2_adam_conv2d_36_kernel_m_read_readvariableop0savev2_adam_conv2d_36_bias_m_read_readvariableop2savev2_adam_conv2d_37_kernel_m_read_readvariableop0savev2_adam_conv2d_37_bias_m_read_readvariableop2savev2_adam_conv2d_38_kernel_m_read_readvariableop0savev2_adam_conv2d_38_bias_m_read_readvariableop2savev2_adam_conv2d_39_kernel_m_read_readvariableop0savev2_adam_conv2d_39_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop2savev2_adam_conv2d_36_kernel_v_read_readvariableop0savev2_adam_conv2d_36_bias_v_read_readvariableop2savev2_adam_conv2d_37_kernel_v_read_readvariableop0savev2_adam_conv2d_37_bias_v_read_readvariableop2savev2_adam_conv2d_38_kernel_v_read_readvariableop0savev2_adam_conv2d_38_bias_v_read_readvariableop2savev2_adam_conv2d_39_kernel_v_read_readvariableop0savev2_adam_conv2d_39_bias_v_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*€
_input_shapesн
к: :	А:: : : : : ::: : : А:А:АА:А:АА:А:
А@А:А::: : : : :	А:::: : : А:А:АА:А:АА:А:
А@А:А:	А:::: : : А:А:АА:А:АА:А:
А@А:А: 2(
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
:А: 
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
:	А: 
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
: А:!!

_output_shapes	
:А:."*
(
_output_shapes
:АА:!#

_output_shapes	
:А:.$*
(
_output_shapes
:АА:!%

_output_shapes	
:А:&&"
 
_output_shapes
:
А@А:!'

_output_shapes	
:А:%(!

_output_shapes
:	А: )
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
:А:.2*
(
_output_shapes
:АА:!3

_output_shapes	
:А:&4"
 
_output_shapes
:
А@А:!5

_output_shapes	
:А:6

_output_shapes
: 
•
Ј
&__inference_CNN3_layer_call_fn_1060514
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

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCall∞
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
:€€€€€€€€€*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_CNN3_layer_call_and_return_conditional_losses_10597732
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€KK
!
_user_specified_name	input_1
»в
и!
#__inference__traced_restore_1061726
file_prefix3
 assignvariableop_dense_19_kernel:	А.
 assignvariableop_1_dense_19_bias:&
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
#assignvariableop_17_dense_18_kernel:
А@А0
!assignvariableop_18_dense_18_bias:	АC
5assignvariableop_19_batch_normalization_9_moving_mean:G
9assignvariableop_20_batch_normalization_9_moving_variance:#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_19_kernel_m:	А6
(assignvariableop_26_adam_dense_19_bias_m:D
6assignvariableop_27_adam_batch_normalization_9_gamma_m:C
5assignvariableop_28_adam_batch_normalization_9_beta_m:E
+assignvariableop_29_adam_conv2d_36_kernel_m: 7
)assignvariableop_30_adam_conv2d_36_bias_m: F
+assignvariableop_31_adam_conv2d_37_kernel_m: А8
)assignvariableop_32_adam_conv2d_37_bias_m:	АG
+assignvariableop_33_adam_conv2d_38_kernel_m:АА8
)assignvariableop_34_adam_conv2d_38_bias_m:	АG
+assignvariableop_35_adam_conv2d_39_kernel_m:АА8
)assignvariableop_36_adam_conv2d_39_bias_m:	А>
*assignvariableop_37_adam_dense_18_kernel_m:
А@А7
(assignvariableop_38_adam_dense_18_bias_m:	А=
*assignvariableop_39_adam_dense_19_kernel_v:	А6
(assignvariableop_40_adam_dense_19_bias_v:D
6assignvariableop_41_adam_batch_normalization_9_gamma_v:C
5assignvariableop_42_adam_batch_normalization_9_beta_v:E
+assignvariableop_43_adam_conv2d_36_kernel_v: 7
)assignvariableop_44_adam_conv2d_36_bias_v: F
+assignvariableop_45_adam_conv2d_37_kernel_v: А8
)assignvariableop_46_adam_conv2d_37_bias_v:	АG
+assignvariableop_47_adam_conv2d_38_kernel_v:АА8
)assignvariableop_48_adam_conv2d_38_bias_v:	АG
+assignvariableop_49_adam_conv2d_39_kernel_v:АА8
)assignvariableop_50_adam_conv2d_39_bias_v:	А>
*assignvariableop_51_adam_dense_18_kernel_v:
А@А7
(assignvariableop_52_adam_dense_18_bias_v:	А
identity_54ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ќ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*Џ
value–BЌ6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЉ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*о
_output_shapesџ
Ў::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_19_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_19_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ґ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6™
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7≥
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_9_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8≤
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_9_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9®
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_36_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10™
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_36_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ђ
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_37_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12™
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_37_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ђ
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_38_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14™
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_38_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ђ
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_39_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16™
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_39_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ђ
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_18_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18©
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_18_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19љ
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_9_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѕ
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_9_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21°
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23£
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24£
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25≤
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_19_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26∞
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_19_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Њ
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_9_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28љ
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_9_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29≥
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_36_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_36_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31≥
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_37_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_37_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33≥
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_38_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_38_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35≥
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_39_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36±
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_39_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37≤
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_18_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38∞
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_18_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39≤
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_19_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40∞
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_19_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Њ
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_9_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42љ
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_9_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43≥
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_36_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_36_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45≥
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_37_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46±
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_37_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47≥
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_38_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_38_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49≥
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_39_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_39_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51≤
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_18_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52∞
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_18_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpм	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53я	
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
г
F
*__inference_lambda_9_layer_call_fn_1061040

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lambda_9_layer_call_and_return_conditional_losses_10592972
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€KK:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ў
G
+__inference_flatten_9_layer_call_fn_1061294

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_10590592
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ч
e
,__inference_dropout_18_layer_call_fn_1061283

inputs
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_10591942
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≠
i
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1058906

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ґ
f
G__inference_dropout_19_layer_call_and_return_conditional_losses_1059155

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
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Хw
€
__inference_call_1022135

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
4sequential_9_dense_18_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_18_biasadd_readvariableop_resource:	А:
'dense_19_matmul_readvariableop_resource:	А6
(dense_19_biasadd_readvariableop_resource:
identityИҐdense_19/BiasAdd/ReadVariableOpҐdense_19/MatMul/ReadVariableOpҐBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_9/batch_normalization_9/ReadVariableOpҐ3sequential_9/batch_normalization_9/ReadVariableOp_1Ґ-sequential_9/conv2d_36/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_36/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_37/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_37/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_38/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_38/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_39/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_39/Conv2D/ReadVariableOpҐ,sequential_9/dense_18/BiasAdd/ReadVariableOpҐ+sequential_9/dense_18/MatMul/ReadVariableOpѓ
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack≥
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1≥
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2л
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_sliceЁ
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpг
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
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¬
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3Џ
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D—
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpд
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2 
sequential_9/conv2d_36/BiasAdd•
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
sequential_9/conv2d_36/Reluс
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPoolџ
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D“
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpе
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2 
sequential_9/conv2d_37/BiasAdd¶
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
sequential_9/conv2d_37/Reluт
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool№
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D“
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpе
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_9/conv2d_38/BiasAdd¶
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_9/conv2d_38/Reluт
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool№
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D“
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpе
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2 
sequential_9/conv2d_39/BiasAdd¶
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
sequential_9/conv2d_39/Reluт
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPoolї
 sequential_9/dropout_18/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 sequential_9/dropout_18/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
sequential_9/flatten_9/Const–
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_18/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2 
sequential_9/flatten_9/Reshape—
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOp„
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/MatMulѕ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOpЏ
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/BiasAddЫ
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/Relu≠
 sequential_9/dropout_19/IdentityIdentity(sequential_9/dense_18/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 sequential_9/dropout_19/Identity©
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp±
dense_19/MatMulMatMul)sequential_9/dropout_19/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/MatMulІ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp•
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/BiasAdd|
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/SoftmaxА
IdentityIdentitydense_19/Softmax:softmax:0 ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2И
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
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ј
і
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1061187

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
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
:€€€€€€€€€KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
Reluѕ
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul‘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
к
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1061289

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
л
H
,__inference_dropout_18_layer_call_fn_1061278

inputs
identity”
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_10590512
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ґ
В
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1061247

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
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
:€€€€€€€€€		А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€		А
 
_user_specified_nameinputs
џ
Е
.__inference_sequential_9_layer_call_fn_1060961

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

unknown_12:	А
identityИҐStatefulPartitionedCallЬ
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
:€€€€€€€€€А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_10593932
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
„
e
,__inference_dropout_19_layer_call_fn_1061353

inputs
identityИҐStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_10591552
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Хw
€
__inference_call_1020120

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
4sequential_9_dense_18_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_18_biasadd_readvariableop_resource:	А:
'dense_19_matmul_readvariableop_resource:	А6
(dense_19_biasadd_readvariableop_resource:
identityИҐdense_19/BiasAdd/ReadVariableOpҐdense_19/MatMul/ReadVariableOpҐBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_9/batch_normalization_9/ReadVariableOpҐ3sequential_9/batch_normalization_9/ReadVariableOp_1Ґ-sequential_9/conv2d_36/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_36/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_37/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_37/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_38/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_38/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_39/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_39/Conv2D/ReadVariableOpҐ,sequential_9/dense_18/BiasAdd/ReadVariableOpҐ+sequential_9/dense_18/MatMul/ReadVariableOpѓ
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack≥
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1≥
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2л
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_sliceЁ
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpг
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
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1¬
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3Џ
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D—
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpд
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2 
sequential_9/conv2d_36/BiasAdd•
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
sequential_9/conv2d_36/Reluс
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPoolџ
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D“
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpе
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2 
sequential_9/conv2d_37/BiasAdd¶
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
sequential_9/conv2d_37/Reluт
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool№
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D“
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpе
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_9/conv2d_38/BiasAdd¶
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_9/conv2d_38/Reluт
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool№
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D“
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpе
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2 
sequential_9/conv2d_39/BiasAdd¶
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
sequential_9/conv2d_39/Reluт
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPoolї
 sequential_9/dropout_18/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 sequential_9/dropout_18/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
sequential_9/flatten_9/Const–
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_18/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2 
sequential_9/flatten_9/Reshape—
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOp„
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/MatMulѕ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOpЏ
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/BiasAddЫ
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/Relu≠
 sequential_9/dropout_19/IdentityIdentity(sequential_9/dense_18/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 sequential_9/dropout_19/Identity©
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp±
dense_19/MatMulMatMul)sequential_9/dropout_19/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/MatMulІ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp•
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/BiasAdd|
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/SoftmaxА
IdentityIdentitydense_19/Softmax:softmax:0 ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2И
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
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
ч
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_1061273

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/y«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
©
Ъ
*__inference_dense_18_layer_call_fn_1061326

inputs
unknown:
А@А
	unknown_0:	А
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_18_layer_call_and_return_conditional_losses_10590782
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А@
 
_user_specified_nameinputs
Ѓm
ц
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060603

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
'dense_18_matmul_readvariableop_resource:
А@А7
(dense_18_biasadd_readvariableop_resource:	А
identityИҐ5batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_9/ReadVariableOpҐ&batch_normalization_9/ReadVariableOp_1Ґ conv2d_36/BiasAdd/ReadVariableOpҐconv2d_36/Conv2D/ReadVariableOpҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ conv2d_37/BiasAdd/ReadVariableOpҐconv2d_37/Conv2D/ReadVariableOpҐ conv2d_38/BiasAdd/ReadVariableOpҐconv2d_38/Conv2D/ReadVariableOpҐ conv2d_39/BiasAdd/ReadVariableOpҐconv2d_39/Conv2D/ReadVariableOpҐdense_18/BiasAdd/ReadVariableOpҐdense_18/MatMul/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpХ
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
lambda_9/strided_slice/stack_2™
lambda_9/strided_sliceStridedSliceinputs%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2
lambda_9/strided_sliceґ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOpЉ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1й
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1з
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3≥
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOpе
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
conv2d_36/Conv2D™
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp∞
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
conv2d_36/Relu 
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPoolі
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_37/Conv2D/ReadVariableOpЁ
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
conv2d_37/Conv2DЂ
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp±
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
conv2d_37/ReluЋ
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPoolµ
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_38/Conv2D/ReadVariableOpЁ
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_38/Conv2DЂ
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp±
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_38/ReluЋ
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolµ
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_39/Conv2D/ReadVariableOpЁ
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
conv2d_39/Conv2DЂ
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp±
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
conv2d_39/ReluЋ
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPoolФ
dropout_18/IdentityIdentity!max_pooling2d_39/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_18/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
flatten_9/ConstЬ
flatten_9/ReshapeReshapedropout_18/Identity:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2
flatten_9/Reshape™
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_18/MatMul/ReadVariableOp£
dense_18/MatMulMatMulflatten_9/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/MatMul®
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_18/BiasAdd/ReadVariableOp¶
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/ReluЖ
dropout_19/IdentityIdentitydense_18/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_19/Identityў
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul–
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulу
IdentityIdentitydropout_19/Identity:output:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 2n
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
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
•
Ш
*__inference_dense_19_layer_call_fn_1061014

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_19_layer_call_and_return_conditional_losses_10596202
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
•u
€
__inference_call_1022063

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
4sequential_9_dense_18_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_18_biasadd_readvariableop_resource:	А:
'dense_19_matmul_readvariableop_resource:	А6
(dense_19_biasadd_readvariableop_resource:
identityИҐdense_19/BiasAdd/ReadVariableOpҐdense_19/MatMul/ReadVariableOpҐBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_9/batch_normalization_9/ReadVariableOpҐ3sequential_9/batch_normalization_9/ReadVariableOp_1Ґ-sequential_9/conv2d_36/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_36/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_37/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_37/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_38/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_38/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_39/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_39/Conv2D/ReadVariableOpҐ,sequential_9/dense_18/BiasAdd/ReadVariableOpҐ+sequential_9/dense_18/MatMul/ReadVariableOpѓ
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack≥
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1≥
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2г
#sequential_9/lambda_9/strided_sliceStridedSliceinputs2sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_sliceЁ
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpг
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
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ї
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3Џ
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
sequential_9/conv2d_36/Conv2D—
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOp№
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2 
sequential_9/conv2d_36/BiasAddЭ
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_9/conv2d_36/Reluй
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPoolџ
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
sequential_9/conv2d_37/Conv2D“
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpЁ
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2 
sequential_9/conv2d_37/BiasAddЮ
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_9/conv2d_37/Reluк
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool№
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
sequential_9/conv2d_38/Conv2D“
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpЁ
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_9/conv2d_38/BiasAddЮ
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_9/conv2d_38/Reluк
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool№
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
sequential_9/conv2d_39/Conv2D“
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpЁ
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А2 
sequential_9/conv2d_39/BiasAddЮ
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*(
_output_shapes
:А		А2
sequential_9/conv2d_39/Reluк
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPool≥
 sequential_9/dropout_18/IdentityIdentity.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*(
_output_shapes
:АА2"
 sequential_9/dropout_18/IdentityН
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
sequential_9/flatten_9/Const»
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_18/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0* 
_output_shapes
:
АА@2 
sequential_9/flatten_9/Reshape—
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOpѕ
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_18/MatMulѕ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOp“
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_18/BiasAddУ
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_9/dense_18/Relu•
 sequential_9/dropout_19/IdentityIdentity(sequential_9/dense_18/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_9/dropout_19/Identity©
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp©
dense_19/MatMulMatMul)sequential_9/dropout_19/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_19/MatMulІ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOpЭ
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_19/BiasAddt
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_19/Softmaxш
IdentityIdentitydense_19/Softmax:softmax:0 ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2И
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
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
Ї
≠
E__inference_dense_18_layer_call_and_return_conditional_losses_1061317

inputs2
matmul_readvariableop_resource:
А@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relu«
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mulћ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А@
 
_user_specified_nameinputs
√
a
E__inference_lambda_9_layer_call_and_return_conditional_losses_1058939

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
strided_slice/stack_2э
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€KK:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
≥И
ќ
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060862
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
'dense_18_matmul_readvariableop_resource:
А@А7
(dense_18_biasadd_readvariableop_resource:	А
identityИҐ$batch_normalization_9/AssignNewValueҐ&batch_normalization_9/AssignNewValue_1Ґ5batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_9/ReadVariableOpҐ&batch_normalization_9/ReadVariableOp_1Ґ conv2d_36/BiasAdd/ReadVariableOpҐconv2d_36/Conv2D/ReadVariableOpҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ conv2d_37/BiasAdd/ReadVariableOpҐconv2d_37/Conv2D/ReadVariableOpҐ conv2d_38/BiasAdd/ReadVariableOpҐconv2d_38/Conv2D/ReadVariableOpҐ conv2d_39/BiasAdd/ReadVariableOpҐconv2d_39/Conv2D/ReadVariableOpҐdense_18/BiasAdd/ReadVariableOpҐdense_18/MatMul/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpХ
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
lambda_9/strided_slice/stack_2≤
lambda_9/strided_sliceStridedSlicelambda_9_input%lambda_9/strided_slice/stack:output:0'lambda_9/strided_slice/stack_1:output:0'lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2
lambda_9/strided_sliceґ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_9/ReadVariableOpЉ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1й
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1х
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3lambda_9/strided_slice:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2(
&batch_normalization_9/FusedBatchNormV3∞
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValueЉ
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1≥
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_36/Conv2D/ReadVariableOpе
conv2d_36/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
conv2d_36/Conv2D™
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp∞
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
conv2d_36/BiasAdd~
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
conv2d_36/Relu 
max_pooling2d_36/MaxPoolMaxPoolconv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPoolі
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_37/Conv2D/ReadVariableOpЁ
conv2d_37/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
conv2d_37/Conv2DЂ
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp±
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
conv2d_37/BiasAdd
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
conv2d_37/ReluЋ
max_pooling2d_37/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPoolµ
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_38/Conv2D/ReadVariableOpЁ
conv2d_38/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_38/Conv2DЂ
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp±
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_38/BiasAdd
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_38/ReluЋ
max_pooling2d_38/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPoolµ
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_39/Conv2D/ReadVariableOpЁ
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
conv2d_39/Conv2DЂ
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp±
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
conv2d_39/BiasAdd
conv2d_39/ReluReluconv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
conv2d_39/ReluЋ
max_pooling2d_39/MaxPoolMaxPoolconv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPooly
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_18/dropout/ConstЄ
dropout_18/dropout/MulMul!max_pooling2d_39/MaxPool:output:0!dropout_18/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_18/dropout/MulЕ
dropout_18/dropout/ShapeShape!max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shapeё
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype021
/dropout_18/dropout/random_uniform/RandomUniformЛ
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_18/dropout/GreaterEqual/yу
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2!
dropout_18/dropout/GreaterEqual©
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout_18/dropout/Castѓ
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_18/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
flatten_9/ConstЬ
flatten_9/ReshapeReshapedropout_18/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2
flatten_9/Reshape™
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_18/MatMul/ReadVariableOp£
dense_18/MatMulMatMulflatten_9/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/MatMul®
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_18/BiasAdd/ReadVariableOp¶
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_18/Reluy
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_19/dropout/Const™
dropout_19/dropout/MulMuldense_18/Relu:activations:0!dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_19/dropout/Mul
dropout_19/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shape÷
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype021
/dropout_19/dropout/random_uniform/RandomUniformЛ
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_19/dropout/GreaterEqual/yл
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
dropout_19/dropout/GreaterEqual°
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_19/dropout/CastІ
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_19/dropout/Mul_1ў
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mul–
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul√
IdentityIdentitydropout_19/dropout/Mul_1:z:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp3^conv2d_36/kernel/Regularizer/Square/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€KK: : : : : : : : : : : : : : 2L
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
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:€€€€€€€€€KK
(
_user_specified_namelambda_9_input
ш
e
G__inference_dropout_19_layer_call_and_return_conditional_losses_1059089

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
®(
П
A__inference_CNN3_layer_call_and_return_conditional_losses_1059773

inputs"
sequential_9_1059726:"
sequential_9_1059728:"
sequential_9_1059730:"
sequential_9_1059732:.
sequential_9_1059734: "
sequential_9_1059736: /
sequential_9_1059738: А#
sequential_9_1059740:	А0
sequential_9_1059742:АА#
sequential_9_1059744:	А0
sequential_9_1059746:АА#
sequential_9_1059748:	А(
sequential_9_1059750:
А@А#
sequential_9_1059752:	А#
dense_19_1059755:	А
dense_19_1059757:
identityИҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpҐ dense_19/StatefulPartitionedCallҐ$sequential_9/StatefulPartitionedCallѕ
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_1059726sequential_9_1059728sequential_9_1059730sequential_9_1059732sequential_9_1059734sequential_9_1059736sequential_9_1059738sequential_9_1059740sequential_9_1059742sequential_9_1059744sequential_9_1059746sequential_9_1059748sequential_9_1059750sequential_9_1059752*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_10593932&
$sequential_9/StatefulPartitionedCall√
 dense_19/StatefulPartitionedCallStatefulPartitionedCall-sequential_9/StatefulPartitionedCall:output:0dense_19_1059755dense_19_1059757*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_19_layer_call_and_return_conditional_losses_10596202"
 dense_19/StatefulPartitionedCall≈
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1059734*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulљ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_9_1059750* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul∞
IdentityIdentity)dense_19/StatefulPartitionedCall:output:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp!^dense_19/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
є

ч
E__inference_dense_19_layer_call_and_return_conditional_losses_1059620

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≠
i
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1058894

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѓ≠
ь
A__inference_CNN3_layer_call_and_return_conditional_losses_1060366
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
4sequential_9_dense_18_matmul_readvariableop_resource:
А@АD
5sequential_9_dense_18_biasadd_readvariableop_resource:	А:
'dense_19_matmul_readvariableop_resource:	А6
(dense_19_biasadd_readvariableop_resource:
identityИҐ2conv2d_36/kernel/Regularizer/Square/ReadVariableOpҐ1dense_18/kernel/Regularizer/Square/ReadVariableOpҐdense_19/BiasAdd/ReadVariableOpҐdense_19/MatMul/ReadVariableOpҐ1sequential_9/batch_normalization_9/AssignNewValueҐ3sequential_9/batch_normalization_9/AssignNewValue_1ҐBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐDsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ1sequential_9/batch_normalization_9/ReadVariableOpҐ3sequential_9/batch_normalization_9/ReadVariableOp_1Ґ-sequential_9/conv2d_36/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_36/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_37/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_37/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_38/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_38/Conv2D/ReadVariableOpҐ-sequential_9/conv2d_39/BiasAdd/ReadVariableOpҐ,sequential_9/conv2d_39/Conv2D/ReadVariableOpҐ,sequential_9/dense_18/BiasAdd/ReadVariableOpҐ+sequential_9/dense_18/MatMul/ReadVariableOpѓ
)sequential_9/lambda_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_9/lambda_9/strided_slice/stack≥
+sequential_9/lambda_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_9/lambda_9/strided_slice/stack_1≥
+sequential_9/lambda_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_9/lambda_9/strided_slice/stack_2м
#sequential_9/lambda_9/strided_sliceStridedSliceinput_12sequential_9/lambda_9/strided_slice/stack:output:04sequential_9/lambda_9/strided_slice/stack_1:output:04sequential_9/lambda_9/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:€€€€€€€€€KK*

begin_mask*
end_mask2%
#sequential_9/lambda_9/strided_sliceЁ
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOpг
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
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1–
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3,sequential_9/lambda_9/strided_slice:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€KK:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<25
3sequential_9/batch_normalization_9/FusedBatchNormV3с
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_9/batch_normalization_9/AssignNewValueэ
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_9/batch_normalization_9/AssignNewValue_1Џ
,sequential_9/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_9/conv2d_36/Conv2D/ReadVariableOpЩ
sequential_9/conv2d_36/Conv2DConv2D7sequential_9/batch_normalization_9/FusedBatchNormV3:y:04sequential_9/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK *
paddingSAME*
strides
2
sequential_9/conv2d_36/Conv2D—
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_9/conv2d_36/BiasAdd/ReadVariableOpд
sequential_9/conv2d_36/BiasAddBiasAdd&sequential_9/conv2d_36/Conv2D:output:05sequential_9/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2 
sequential_9/conv2d_36/BiasAdd•
sequential_9/conv2d_36/ReluRelu'sequential_9/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€KK 2
sequential_9/conv2d_36/Reluс
%sequential_9/max_pooling2d_36/MaxPoolMaxPool)sequential_9/conv2d_36/Relu:activations:0*/
_output_shapes
:€€€€€€€€€%% *
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_36/MaxPoolџ
,sequential_9/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_37_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_9/conv2d_37/Conv2D/ReadVariableOpС
sequential_9/conv2d_37/Conv2DConv2D.sequential_9/max_pooling2d_36/MaxPool:output:04sequential_9/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А*
paddingSAME*
strides
2
sequential_9/conv2d_37/Conv2D“
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_37/BiasAdd/ReadVariableOpе
sequential_9/conv2d_37/BiasAddBiasAdd&sequential_9/conv2d_37/Conv2D:output:05sequential_9/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2 
sequential_9/conv2d_37/BiasAdd¶
sequential_9/conv2d_37/ReluRelu'sequential_9/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€%%А2
sequential_9/conv2d_37/Reluт
%sequential_9/max_pooling2d_37/MaxPoolMaxPool)sequential_9/conv2d_37/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_37/MaxPool№
,sequential_9/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_38/Conv2D/ReadVariableOpС
sequential_9/conv2d_38/Conv2DConv2D.sequential_9/max_pooling2d_37/MaxPool:output:04sequential_9/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_9/conv2d_38/Conv2D“
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_38/BiasAdd/ReadVariableOpе
sequential_9/conv2d_38/BiasAddBiasAdd&sequential_9/conv2d_38/Conv2D:output:05sequential_9/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_9/conv2d_38/BiasAdd¶
sequential_9/conv2d_38/ReluRelu'sequential_9/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_9/conv2d_38/Reluт
%sequential_9/max_pooling2d_38/MaxPoolMaxPool)sequential_9/conv2d_38/Relu:activations:0*0
_output_shapes
:€€€€€€€€€		А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_38/MaxPool№
,sequential_9/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_39_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_39/Conv2D/ReadVariableOpС
sequential_9/conv2d_39/Conv2DConv2D.sequential_9/max_pooling2d_38/MaxPool:output:04sequential_9/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А*
paddingSAME*
strides
2
sequential_9/conv2d_39/Conv2D“
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_39/BiasAdd/ReadVariableOpе
sequential_9/conv2d_39/BiasAddBiasAdd&sequential_9/conv2d_39/Conv2D:output:05sequential_9/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€		А2 
sequential_9/conv2d_39/BiasAdd¶
sequential_9/conv2d_39/ReluRelu'sequential_9/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€		А2
sequential_9/conv2d_39/Reluт
%sequential_9/max_pooling2d_39/MaxPoolMaxPool)sequential_9/conv2d_39/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_9/max_pooling2d_39/MaxPoolУ
%sequential_9/dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2'
%sequential_9/dropout_18/dropout/Constм
#sequential_9/dropout_18/dropout/MulMul.sequential_9/max_pooling2d_39/MaxPool:output:0.sequential_9/dropout_18/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2%
#sequential_9/dropout_18/dropout/Mulђ
%sequential_9/dropout_18/dropout/ShapeShape.sequential_9/max_pooling2d_39/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_18/dropout/ShapeЕ
<sequential_9/dropout_18/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_18/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype02>
<sequential_9/dropout_18/dropout/random_uniform/RandomUniform•
.sequential_9/dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=20
.sequential_9/dropout_18/dropout/GreaterEqual/yІ
,sequential_9/dropout_18/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_18/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_18/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2.
,sequential_9/dropout_18/dropout/GreaterEqual–
$sequential_9/dropout_18/dropout/CastCast0sequential_9/dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2&
$sequential_9/dropout_18/dropout/Castг
%sequential_9/dropout_18/dropout/Mul_1Mul'sequential_9/dropout_18/dropout/Mul:z:0(sequential_9/dropout_18/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2'
%sequential_9/dropout_18/dropout/Mul_1Н
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
sequential_9/flatten_9/Const–
sequential_9/flatten_9/ReshapeReshape)sequential_9/dropout_18/dropout/Mul_1:z:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А@2 
sequential_9/flatten_9/Reshape—
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOp„
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/MatMulѕ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOpЏ
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/BiasAddЫ
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_18/ReluУ
%sequential_9/dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_9/dropout_19/dropout/Constё
#sequential_9/dropout_19/dropout/MulMul(sequential_9/dense_18/Relu:activations:0.sequential_9/dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2%
#sequential_9/dropout_19/dropout/Mul¶
%sequential_9/dropout_19/dropout/ShapeShape(sequential_9/dense_18/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dropout_19/dropout/Shapeэ
<sequential_9/dropout_19/dropout/random_uniform/RandomUniformRandomUniform.sequential_9/dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02>
<sequential_9/dropout_19/dropout/random_uniform/RandomUniform•
.sequential_9/dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_9/dropout_19/dropout/GreaterEqual/yЯ
,sequential_9/dropout_19/dropout/GreaterEqualGreaterEqualEsequential_9/dropout_19/dropout/random_uniform/RandomUniform:output:07sequential_9/dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2.
,sequential_9/dropout_19/dropout/GreaterEqual»
$sequential_9/dropout_19/dropout/CastCast0sequential_9/dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2&
$sequential_9/dropout_19/dropout/Castџ
%sequential_9/dropout_19/dropout/Mul_1Mul'sequential_9/dropout_19/dropout/Mul:z:0(sequential_9/dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%sequential_9/dropout_19/dropout/Mul_1©
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp±
dense_19/MatMulMatMul)sequential_9/dropout_19/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/MatMulІ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp•
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/BiasAdd|
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_19/Softmaxж
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_9_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_36/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_36/kernel/Regularizer/SquareSquare:conv2d_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_36/kernel/Regularizer/Square°
"conv2d_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_36/kernel/Regularizer/Const¬
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
„#<2$
"conv2d_36/kernel/Regularizer/mul/xƒ
 conv2d_36/kernel/Regularizer/mulMul+conv2d_36/kernel/Regularizer/mul/x:output:0)conv2d_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_36/kernel/Regularizer/mulЁ
1dense_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_18/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_18/kernel/Regularizer/SquareSquare9dense_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_18/kernel/Regularizer/SquareЧ
!dense_18/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_18/kernel/Regularizer/ConstЊ
dense_18/kernel/Regularizer/SumSum&dense_18/kernel/Regularizer/Square:y:0*dense_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/SumЛ
!dense_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_18/kernel/Regularizer/mul/xј
dense_18/kernel/Regularizer/mulMul*dense_18/kernel/Regularizer/mul/x:output:0(dense_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_18/kernel/Regularizer/mul”
IdentityIdentitydense_19/Softmax:softmax:03^conv2d_36/kernel/Regularizer/Square/ReadVariableOp2^dense_18/kernel/Regularizer/Square/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1.^sequential_9/conv2d_36/BiasAdd/ReadVariableOp-^sequential_9/conv2d_36/Conv2D/ReadVariableOp.^sequential_9/conv2d_37/BiasAdd/ReadVariableOp-^sequential_9/conv2d_37/Conv2D/ReadVariableOp.^sequential_9/conv2d_38/BiasAdd/ReadVariableOp-^sequential_9/conv2d_38/Conv2D/ReadVariableOp.^sequential_9/conv2d_39/BiasAdd/ReadVariableOp-^sequential_9/conv2d_39/Conv2D/ReadVariableOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:€€€€€€€€€KK: : : : : : : : : : : : : : : : 2h
2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2conv2d_36/kernel/Regularizer/Square/ReadVariableOp2f
1dense_18/kernel/Regularizer/Square/ReadVariableOp1dense_18/kernel/Regularizer/Square/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2f
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
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€KK
!
_user_specified_name	input_1
¶
“
7__inference_batch_normalization_9_layer_call_fn_1061164

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€KK*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_10592702
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€KK: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€KK
 
_user_specified_nameinputs
М
Э
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061058

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≥
serving_defaultЯ
C
input_18
serving_default_input_1:0€€€€€€€€€KK<
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ўК
Б

h2ptjl
_output
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
й_default_save_signature
+к&call_and_return_all_conditional_losses
л__call__
	мcall"П	
_tf_keras_modelх{"name": "CNN3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN3", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN3"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
∞m
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
trainable_variables
	variables
regularization_losses
	keras_api
+н&call_and_return_all_conditional_losses
о__call__"…i
_tf_keras_sequential™i{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_9_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 33, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_9_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_9_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 26}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 32}]}}}
„

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
+п&call_and_return_all_conditional_losses
р__call__"∞
_tf_keras_layerЦ{"name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
л
!iter

"beta_1

#beta_2
	$decay
%learning_ratemЌmќ&mѕ'm–(m—)m“*m”+m‘,m’-m÷.m„/mЎ0mў1mЏvџv№&vЁ'vё(vя)vа*vб+vв,vг-vд.vе/vж0vз1vи"
	optimizer
Ж
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
Ц
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
ќ
trainable_variables
4non_trainable_variables
	variables
regularization_losses
5layer_regularization_losses
6layer_metrics

7layers
8metrics
л__call__
й_default_save_signature
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
-
сserving_default"
signature_map
Ў
9trainable_variables
:	variables
;regularization_losses
<	keras_api
+т&call_and_return_all_conditional_losses
у__call__"«
_tf_keras_layer≠{"name": "lambda_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_9", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
ƒ

=axis
	&gamma
'beta
2moving_mean
3moving_variance
>trainable_variables
?	variables
@regularization_losses
A	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"о
_tf_keras_layer‘{"name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
Ґ

(kernel
)bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"ы	
_tf_keras_layerб	{"name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
≥
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 40}}
÷


*kernel
+bias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
+ъ&call_and_return_all_conditional_losses
ы__call__"ѓ	
_tf_keras_layerХ	{"name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
≥
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
+ь&call_and_return_all_conditional_losses
э__call__"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}}
Ў


,kernel
-bias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
+ю&call_and_return_all_conditional_losses
€__call__"±	
_tf_keras_layerЧ	{"name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
≥
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 44}}
÷


.kernel
/bias
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"ѓ	
_tf_keras_layerХ	{"name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 9, 9, 256]}}
≥
^trainable_variables
_	variables
`regularization_losses
a	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 46}}
Б
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"р
_tf_keras_layer÷{"name": "dropout_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 26}
Ш
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"З
_tf_keras_layerн{"name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 47}}
®	

0kernel
1bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"Б
_tf_keras_layerз{"name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 8192]}}
Б
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"р
_tf_keras_layer÷{"name": "dropout_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 32}
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
Ж
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
0
О0
П1"
trackable_list_wrapper
∞
trainable_variables
rnon_trainable_variables
	variables
regularization_losses
slayer_regularization_losses
tlayer_metrics

ulayers
vmetrics
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_19/kernel
:2dense_19/bias
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
∞
trainable_variables
wnon_trainable_variables
	variables
regularization_losses
xlayer_regularization_losses
ylayer_metrics

zlayers
{metrics
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
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
А@А2dense_18/kernel
:А2dense_18/bias
1:/ (2!batch_normalization_9/moving_mean
5:3 (2%batch_normalization_9/moving_variance
.
20
31"
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
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≥
9trainable_variables
~non_trainable_variables
:	variables
;regularization_losses
layer_regularization_losses
Аlayer_metrics
Бlayers
Вmetrics
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
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
µ
>trainable_variables
Гnon_trainable_variables
?	variables
@regularization_losses
 Дlayer_regularization_losses
Еlayer_metrics
Жlayers
Зmetrics
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
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
О0"
trackable_list_wrapper
µ
Btrainable_variables
Иnon_trainable_variables
C	variables
Dregularization_losses
 Йlayer_regularization_losses
Кlayer_metrics
Лlayers
Мmetrics
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ftrainable_variables
Нnon_trainable_variables
G	variables
Hregularization_losses
 Оlayer_regularization_losses
Пlayer_metrics
Рlayers
Сmetrics
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
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
µ
Jtrainable_variables
Тnon_trainable_variables
K	variables
Lregularization_losses
 Уlayer_regularization_losses
Фlayer_metrics
Хlayers
Цmetrics
ы__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ntrainable_variables
Чnon_trainable_variables
O	variables
Pregularization_losses
 Шlayer_regularization_losses
Щlayer_metrics
Ъlayers
Ыmetrics
э__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
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
µ
Rtrainable_variables
Ьnon_trainable_variables
S	variables
Tregularization_losses
 Эlayer_regularization_losses
Юlayer_metrics
Яlayers
†metrics
€__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Vtrainable_variables
°non_trainable_variables
W	variables
Xregularization_losses
 Ґlayer_regularization_losses
£layer_metrics
§layers
•metrics
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
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
µ
Ztrainable_variables
¶non_trainable_variables
[	variables
\regularization_losses
 Іlayer_regularization_losses
®layer_metrics
©layers
™metrics
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
^trainable_variables
Ђnon_trainable_variables
_	variables
`regularization_losses
 ђlayer_regularization_losses
≠layer_metrics
Ѓlayers
ѓmetrics
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
btrainable_variables
∞non_trainable_variables
c	variables
dregularization_losses
 ±layer_regularization_losses
≤layer_metrics
≥layers
іmetrics
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ftrainable_variables
µnon_trainable_variables
g	variables
hregularization_losses
 ґlayer_regularization_losses
Јlayer_metrics
Єlayers
єmetrics
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
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
П0"
trackable_list_wrapper
µ
jtrainable_variables
Їnon_trainable_variables
k	variables
lregularization_losses
 їlayer_regularization_losses
Љlayer_metrics
љlayers
Њmetrics
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ntrainable_variables
њnon_trainable_variables
o	variables
pregularization_losses
 јlayer_regularization_losses
Ѕlayer_metrics
¬layers
√metrics
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ж
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
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ў

ƒtotal

≈count
∆	variables
«	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
Ы

»total

…count
 
_fn_kwargs
Ћ	variables
ћ	keras_api"ѕ
_tf_keras_metricі{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
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
20
31"
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
О0"
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
П0"
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
ƒ0
≈1"
trackable_list_wrapper
.
∆	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
»0
…1"
trackable_list_wrapper
.
Ћ	variables"
_generic_user_object
':%	А2Adam/dense_19/kernel/m
 :2Adam/dense_19/bias/m
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
А@А2Adam/dense_18/kernel/m
!:А2Adam/dense_18/bias/m
':%	А2Adam/dense_19/kernel/v
 :2Adam/dense_19/bias/v
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
А@А2Adam/dense_18/kernel/v
!:А2Adam/dense_18/bias/v
и2е
"__inference__wrapped_model_1058750Њ
Л≤З
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
annotations™ *.Ґ+
)К&
input_1€€€€€€€€€KK
∆2√
A__inference_CNN3_layer_call_and_return_conditional_losses_1060086
A__inference_CNN3_layer_call_and_return_conditional_losses_1060184
A__inference_CNN3_layer_call_and_return_conditional_losses_1060268
A__inference_CNN3_layer_call_and_return_conditional_losses_1060366і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
Џ2„
&__inference_CNN3_layer_call_fn_1060403
&__inference_CNN3_layer_call_fn_1060440
&__inference_CNN3_layer_call_fn_1060477
&__inference_CNN3_layer_call_fn_1060514і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
З2Д
__inference_call_1021991
__inference_call_1022063
__inference_call_1022135≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060603
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060694
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060771
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060862ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
Ж2Г
.__inference_sequential_9_layer_call_fn_1060895
.__inference_sequential_9_layer_call_fn_1060928
.__inference_sequential_9_layer_call_fn_1060961
.__inference_sequential_9_layer_call_fn_1060994ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
п2м
E__inference_dense_19_layer_call_and_return_conditional_losses_1061005Ґ
Щ≤Х
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
annotations™ *
 
‘2—
*__inference_dense_19_layer_call_fn_1061014Ґ
Щ≤Х
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
annotations™ *
 
ћB…
%__inference_signature_wrapper_1060002input_1"Ф
Н≤Й
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
annotations™ *
 
‘2—
E__inference_lambda_9_layer_call_and_return_conditional_losses_1061022
E__inference_lambda_9_layer_call_and_return_conditional_losses_1061030ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
*__inference_lambda_9_layer_call_fn_1061035
*__inference_lambda_9_layer_call_fn_1061040ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
К2З
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061058
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061076
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061094
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061112і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
7__inference_batch_normalization_9_layer_call_fn_1061125
7__inference_batch_normalization_9_layer_call_fn_1061138
7__inference_batch_normalization_9_layer_call_fn_1061151
7__inference_batch_normalization_9_layer_call_fn_1061164і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
р2н
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1061187Ґ
Щ≤Х
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
annotations™ *
 
’2“
+__inference_conv2d_36_layer_call_fn_1061196Ґ
Щ≤Х
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
annotations™ *
 
µ2≤
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1058882а
Щ≤Х
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
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ2Ч
2__inference_max_pooling2d_36_layer_call_fn_1058888а
Щ≤Х
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
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
р2н
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1061207Ґ
Щ≤Х
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
annotations™ *
 
’2“
+__inference_conv2d_37_layer_call_fn_1061216Ґ
Щ≤Х
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
annotations™ *
 
µ2≤
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1058894а
Щ≤Х
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
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ2Ч
2__inference_max_pooling2d_37_layer_call_fn_1058900а
Щ≤Х
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
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
р2н
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1061227Ґ
Щ≤Х
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
annotations™ *
 
’2“
+__inference_conv2d_38_layer_call_fn_1061236Ґ
Щ≤Х
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
annotations™ *
 
µ2≤
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1058906а
Щ≤Х
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
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ2Ч
2__inference_max_pooling2d_38_layer_call_fn_1058912а
Щ≤Х
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
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
р2н
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1061247Ґ
Щ≤Х
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
annotations™ *
 
’2“
+__inference_conv2d_39_layer_call_fn_1061256Ґ
Щ≤Х
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
annotations™ *
 
µ2≤
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1058918а
Щ≤Х
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
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ2Ч
2__inference_max_pooling2d_39_layer_call_fn_1058924а
Щ≤Х
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
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ћ2…
G__inference_dropout_18_layer_call_and_return_conditional_losses_1061261
G__inference_dropout_18_layer_call_and_return_conditional_losses_1061273і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
Ц2У
,__inference_dropout_18_layer_call_fn_1061278
,__inference_dropout_18_layer_call_fn_1061283і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
р2н
F__inference_flatten_9_layer_call_and_return_conditional_losses_1061289Ґ
Щ≤Х
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
annotations™ *
 
’2“
+__inference_flatten_9_layer_call_fn_1061294Ґ
Щ≤Х
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
annotations™ *
 
п2м
E__inference_dense_18_layer_call_and_return_conditional_losses_1061317Ґ
Щ≤Х
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
annotations™ *
 
‘2—
*__inference_dense_18_layer_call_fn_1061326Ґ
Щ≤Х
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
annotations™ *
 
ћ2…
G__inference_dropout_19_layer_call_and_return_conditional_losses_1061331
G__inference_dropout_19_layer_call_and_return_conditional_losses_1061343і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
Ц2У
,__inference_dropout_19_layer_call_fn_1061348
,__inference_dropout_19_layer_call_fn_1061353і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
і2±
__inference_loss_fn_0_1061364П
З≤Г
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
annotations™ *Ґ 
і2±
__inference_loss_fn_1_1061375П
З≤Г
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
annotations™ *Ґ ї
A__inference_CNN3_layer_call_and_return_conditional_losses_1060086v&'23()*+,-./01;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ї
A__inference_CNN3_layer_call_and_return_conditional_losses_1060184v&'23()*+,-./01;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ Љ
A__inference_CNN3_layer_call_and_return_conditional_losses_1060268w&'23()*+,-./01<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€KK
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Љ
A__inference_CNN3_layer_call_and_return_conditional_losses_1060366w&'23()*+,-./01<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€KK
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ф
&__inference_CNN3_layer_call_fn_1060403j&'23()*+,-./01<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€KK
p 
™ "К€€€€€€€€€У
&__inference_CNN3_layer_call_fn_1060440i&'23()*+,-./01;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p 
™ "К€€€€€€€€€У
&__inference_CNN3_layer_call_fn_1060477i&'23()*+,-./01;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p
™ "К€€€€€€€€€Ф
&__inference_CNN3_layer_call_fn_1060514j&'23()*+,-./01<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€KK
p
™ "К€€€€€€€€€®
"__inference__wrapped_model_1058750Б&'23()*+,-./018Ґ5
.Ґ+
)К&
input_1€€€€€€€€€KK
™ "3™0
.
output_1"К
output_1€€€€€€€€€н
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061058Ц&'23MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ н
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061076Ц&'23MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ »
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061094r&'23;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p 
™ "-Ґ*
#К 
0€€€€€€€€€KK
Ъ »
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1061112r&'23;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p
™ "-Ґ*
#К 
0€€€€€€€€€KK
Ъ ≈
7__inference_batch_normalization_9_layer_call_fn_1061125Й&'23MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€≈
7__inference_batch_normalization_9_layer_call_fn_1061138Й&'23MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€†
7__inference_batch_normalization_9_layer_call_fn_1061151e&'23;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p 
™ " К€€€€€€€€€KK†
7__inference_batch_normalization_9_layer_call_fn_1061164e&'23;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p
™ " К€€€€€€€€€KKu
__inference_call_1021991Y&'23()*+,-./013Ґ0
)Ґ&
 К
inputsАKK
p
™ "К	Аu
__inference_call_1022063Y&'23()*+,-./013Ґ0
)Ґ&
 К
inputsАKK
p 
™ "К	АЕ
__inference_call_1022135i&'23()*+,-./01;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€KK
p 
™ "К€€€€€€€€€ґ
F__inference_conv2d_36_layer_call_and_return_conditional_losses_1061187l()7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€KK
™ "-Ґ*
#К 
0€€€€€€€€€KK 
Ъ О
+__inference_conv2d_36_layer_call_fn_1061196_()7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€KK
™ " К€€€€€€€€€KK Ј
F__inference_conv2d_37_layer_call_and_return_conditional_losses_1061207m*+7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€%% 
™ ".Ґ+
$К!
0€€€€€€€€€%%А
Ъ П
+__inference_conv2d_37_layer_call_fn_1061216`*+7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€%% 
™ "!К€€€€€€€€€%%АЄ
F__inference_conv2d_38_layer_call_and_return_conditional_losses_1061227n,-8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Р
+__inference_conv2d_38_layer_call_fn_1061236a,-8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€АЄ
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1061247n./8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€		А
™ ".Ґ+
$К!
0€€€€€€€€€		А
Ъ Р
+__inference_conv2d_39_layer_call_fn_1061256a./8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€		А
™ "!К€€€€€€€€€		АІ
E__inference_dense_18_layer_call_and_return_conditional_losses_1061317^010Ґ-
&Ґ#
!К
inputs€€€€€€€€€А@
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
*__inference_dense_18_layer_call_fn_1061326Q010Ґ-
&Ґ#
!К
inputs€€€€€€€€€А@
™ "К€€€€€€€€€А¶
E__inference_dense_19_layer_call_and_return_conditional_losses_1061005]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
*__inference_dense_19_layer_call_fn_1061014P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€є
G__inference_dropout_18_layer_call_and_return_conditional_losses_1061261n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ є
G__inference_dropout_18_layer_call_and_return_conditional_losses_1061273n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ С
,__inference_dropout_18_layer_call_fn_1061278a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€АС
,__inference_dropout_18_layer_call_fn_1061283a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€А©
G__inference_dropout_19_layer_call_and_return_conditional_losses_1061331^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ©
G__inference_dropout_19_layer_call_and_return_conditional_losses_1061343^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ Б
,__inference_dropout_19_layer_call_fn_1061348Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€АБ
,__inference_dropout_19_layer_call_fn_1061353Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€Ађ
F__inference_flatten_9_layer_call_and_return_conditional_losses_1061289b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А@
Ъ Д
+__inference_flatten_9_layer_call_fn_1061294U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€А@є
E__inference_lambda_9_layer_call_and_return_conditional_losses_1061022p?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€KK

 
p 
™ "-Ґ*
#К 
0€€€€€€€€€KK
Ъ є
E__inference_lambda_9_layer_call_and_return_conditional_losses_1061030p?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€KK

 
p
™ "-Ґ*
#К 
0€€€€€€€€€KK
Ъ С
*__inference_lambda_9_layer_call_fn_1061035c?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€KK

 
p 
™ " К€€€€€€€€€KKС
*__inference_lambda_9_layer_call_fn_1061040c?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€KK

 
p
™ " К€€€€€€€€€KK<
__inference_loss_fn_0_1061364(Ґ

Ґ 
™ "К <
__inference_loss_fn_1_10613750Ґ

Ґ 
™ "К р
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_1058882ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ »
2__inference_max_pooling2d_36_layer_call_fn_1058888СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€р
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1058894ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ »
2__inference_max_pooling2d_37_layer_call_fn_1058900СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€р
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1058906ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ »
2__inference_max_pooling2d_38_layer_call_fn_1058912СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€р
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1058918ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ »
2__inference_max_pooling2d_39_layer_call_fn_1058924СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€∆
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060603y&'23()*+,-./01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€KK
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ∆
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060694y&'23()*+,-./01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€KK
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ѕ
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060771Б&'23()*+,-./01GҐD
=Ґ:
0К-
lambda_9_input€€€€€€€€€KK
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ѕ
I__inference_sequential_9_layer_call_and_return_conditional_losses_1060862Б&'23()*+,-./01GҐD
=Ґ:
0К-
lambda_9_input€€€€€€€€€KK
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ¶
.__inference_sequential_9_layer_call_fn_1060895t&'23()*+,-./01GҐD
=Ґ:
0К-
lambda_9_input€€€€€€€€€KK
p 

 
™ "К€€€€€€€€€АЮ
.__inference_sequential_9_layer_call_fn_1060928l&'23()*+,-./01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€KK
p 

 
™ "К€€€€€€€€€АЮ
.__inference_sequential_9_layer_call_fn_1060961l&'23()*+,-./01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€KK
p

 
™ "К€€€€€€€€€А¶
.__inference_sequential_9_layer_call_fn_1060994t&'23()*+,-./01GҐD
=Ґ:
0К-
lambda_9_input€€€€€€€€€KK
p

 
™ "К€€€€€€€€€Аґ
%__inference_signature_wrapper_1060002М&'23()*+,-./01CҐ@
Ґ 
9™6
4
input_1)К&
input_1€€€€€€€€€KK"3™0
.
output_1"К
output_1€€€€€€€€€