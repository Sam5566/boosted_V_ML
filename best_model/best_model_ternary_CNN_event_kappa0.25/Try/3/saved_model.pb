хн
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
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718═П
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	А*
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
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
В
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
Е
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*!
shared_nameconv2d_10/kernel
~
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*'
_output_shapes
: А*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:А*
dtype0
{
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*
shared_namedense_9/kernel
t
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*!
_output_shapes
:АвА*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:А*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
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
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_11/kernel/m
В
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes
:	А*
dtype0
А
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
Ь
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/m
Х
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/m
У
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:*
dtype0
Р
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_9/kernel/m
Й
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
_output_shapes
: *
dtype0
А
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
У
Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_10/kernel/m
М
+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_10/bias/m
|
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_11/kernel/m
Н
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_11/bias/m
|
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*&
shared_nameAdam/dense_9/kernel/m
В
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*!
_output_shapes
:АвА*
dtype0

Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_9/bias/m
x
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_10/kernel/m
Г
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_10/bias/m
z
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_11/kernel/v
В
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes
:	А*
dtype0
А
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
Ь
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/v
Х
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/v
У
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:*
dtype0
Р
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_9/kernel/v
Й
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
: *
dtype0
А
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
У
Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_10/kernel/v
М
+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_10/bias/v
|
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_11/kernel/v
Н
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_11/bias/v
|
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes	
:А*
dtype0
Й
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*&
shared_nameAdam/dense_9/kernel/v
В
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*!
_output_shapes
:АвА*
dtype0

Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_9/bias/v
x
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_10/kernel/v
Г
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_10/bias/v
z
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
▄]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ч]
valueН]BК] BГ]
К

h2ptjl
_output
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
и
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
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
╪
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m╧'m╨*m╤+m╥,m╙-m╘.m╒/m╓0m╫1m╪2m┘3m┌v█v▄&v▌'v▐*v▀+vр,vс-vт.vу/vф0vх1vц2vч3vш
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
 
н
4layer_metrics
	variables
5layer_regularization_losses
6metrics
7non_trainable_variables
trainable_variables

8layers
regularization_losses
 
R
9	variables
:regularization_losses
;trainable_variables
<	keras_api
Ч
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
 
н
rlayer_metrics
	variables
slayer_regularization_losses
tmetrics
unon_trainable_variables
trainable_variables

vlayers
regularization_losses
NL
VARIABLE_VALUEdense_11/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_11/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
wlayer_metrics
	variables
xmetrics
ynon_trainable_variables
regularization_losses
trainable_variables

zlayers
{layer_regularization_losses
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
VARIABLE_VALUEbatch_normalization_3/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_3/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_3/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_3/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_9/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_9/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_10/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_10/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_11/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_11/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_9/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_9/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_10/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_10/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 
 

|0
}1

(0
)1

0
1
 
 
 
░
~layer_metrics
9	variables
metrics
Аnon_trainable_variables
:regularization_losses
;trainable_variables
Бlayers
 Вlayer_regularization_losses
 

&0
'1
(2
)3
 

&0
'1
▓
Гlayer_metrics
>	variables
Дmetrics
Еnon_trainable_variables
?regularization_losses
@trainable_variables
Жlayers
 Зlayer_regularization_losses

*0
+1
 

*0
+1
▓
Иlayer_metrics
B	variables
Йmetrics
Кnon_trainable_variables
Cregularization_losses
Dtrainable_variables
Лlayers
 Мlayer_regularization_losses
 
 
 
▓
Нlayer_metrics
F	variables
Оmetrics
Пnon_trainable_variables
Gregularization_losses
Htrainable_variables
Рlayers
 Сlayer_regularization_losses

,0
-1
 

,0
-1
▓
Тlayer_metrics
J	variables
Уmetrics
Фnon_trainable_variables
Kregularization_losses
Ltrainable_variables
Хlayers
 Цlayer_regularization_losses
 
 
 
▓
Чlayer_metrics
N	variables
Шmetrics
Щnon_trainable_variables
Oregularization_losses
Ptrainable_variables
Ъlayers
 Ыlayer_regularization_losses

.0
/1
 

.0
/1
▓
Ьlayer_metrics
R	variables
Эmetrics
Юnon_trainable_variables
Sregularization_losses
Ttrainable_variables
Яlayers
 аlayer_regularization_losses
 
 
 
▓
бlayer_metrics
V	variables
вmetrics
гnon_trainable_variables
Wregularization_losses
Xtrainable_variables
дlayers
 еlayer_regularization_losses
 
 
 
▓
жlayer_metrics
Z	variables
зmetrics
иnon_trainable_variables
[regularization_losses
\trainable_variables
йlayers
 кlayer_regularization_losses
 
 
 
▓
лlayer_metrics
^	variables
мmetrics
нnon_trainable_variables
_regularization_losses
`trainable_variables
оlayers
 пlayer_regularization_losses

00
11
 

00
11
▓
░layer_metrics
b	variables
▒metrics
▓non_trainable_variables
cregularization_losses
dtrainable_variables
│layers
 ┤layer_regularization_losses
 
 
 
▓
╡layer_metrics
f	variables
╢metrics
╖non_trainable_variables
gregularization_losses
htrainable_variables
╕layers
 ╣layer_regularization_losses

20
31
 

20
31
▓
║layer_metrics
j	variables
╗metrics
╝non_trainable_variables
kregularization_losses
ltrainable_variables
╜layers
 ╛layer_regularization_losses
 
 
 
▓
┐layer_metrics
n	variables
└metrics
┴non_trainable_variables
oregularization_losses
ptrainable_variables
┬layers
 ├layer_regularization_losses
 
 
 

(0
)1
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
8

─total

┼count
╞	variables
╟	keras_api
I

╚total

╔count
╩
_fn_kwargs
╦	variables
╠	keras_api
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
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

─0
┼1

╞	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

╚0
╔1

╦	variables
qo
VARIABLE_VALUEAdam/dense_11/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_11/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_9/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_9/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_10/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_10/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_11/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_11/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_9/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_9/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_10/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_10/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_11/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_11/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_9/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_9/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_10/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_10/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_11/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_11/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_9/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_9/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_10/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_10/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
Ь
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *-
f(R&
$__inference_signature_wrapper_570819
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
▌
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOpConst*B
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
GPU2 *0J 8В *(
f#R!
__inference__traced_save_572506
┤
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biastotalcounttotal_1count_1Adam/dense_11/kernel/mAdam/dense_11/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/vAdam/dense_11/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/v*A
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
GPU2 *0J 8В *+
f&R$
"__inference__traced_restore_572675нб
╚
Ю
)__inference_conv2d_9_layer_call_fn_572078

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallБ
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
GPU2 *0J 8В *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_5697132
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
иu
ы
__inference_call_489777

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
3sequential_3_dense_9_matmul_readvariableop_resource:АвАC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А6
(dense_11_biasadd_readvariableop_resource:
identityИвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpвBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_3/batch_normalization_3/ReadVariableOpв3sequential_3/batch_normalization_3/ReadVariableOp_1в-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpп
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack│
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1│
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2ы
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice▌
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpу
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1Р
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3╫
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOpЦ
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D╬
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpр
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/BiasAddв
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/Reluю
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool█
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpР
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D╥
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpх
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_3/conv2d_10/BiasAddж
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_3/conv2d_10/ReluЄ
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool▄
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpС
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D╥
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpх
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_3/conv2d_11/BiasAddж
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_3/conv2d_11/ReluЄ
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool╣
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:         		А2!
sequential_3/dropout_9/IdentityН
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_3/flatten_3/Const╨
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2 
sequential_3/flatten_3/Reshape╧
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp╘
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/MatMul╠
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp╓
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/BiasAddШ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/Reluм
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_3/dropout_10/Identity╤
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp┘
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/MatMul╧
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp┌
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/BiasAddЫ
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/Reluн
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_3/dropout_11/Identityй
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOp▒
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/MatMulз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpе
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_11/Softmax·
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2И
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
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
:         KK
 
_user_specified_nameinputs
■
о
$__inference_signature_wrapper_570819
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

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallТ
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
:         *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В **
f%R#
!__inference__wrapped_model_5694902
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
°└
Ш
?__inference_CNN_layer_call_and_return_conditional_losses_571168

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
3sequential_3_dense_9_matmul_readvariableop_resource:АвАC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А6
(dense_11_biasadd_readvariableop_resource:
identityИв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpв1sequential_3/batch_normalization_3/AssignNewValueв3sequential_3/batch_normalization_3/AssignNewValue_1вBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_3/batch_normalization_3/ReadVariableOpв3sequential_3/batch_normalization_3/ReadVariableOp_1в-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpп
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack│
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1│
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2ы
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice▌
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpу
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1Р
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1╨
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<25
3sequential_3/batch_normalization_3/FusedBatchNormV3ё
1sequential_3/batch_normalization_3/AssignNewValueAssignVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource@sequential_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_3/batch_normalization_3/AssignNewValue¤
3sequential_3/batch_normalization_3/AssignNewValue_1AssignVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceDsequential_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0E^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_3/batch_normalization_3/AssignNewValue_1╫
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOpЦ
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D╬
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpр
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/BiasAddв
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/Reluю
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool█
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpР
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D╥
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpх
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_3/conv2d_10/BiasAddж
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_3/conv2d_10/ReluЄ
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool▄
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpС
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D╥
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpх
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_3/conv2d_11/BiasAddж
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_3/conv2d_11/ReluЄ
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPoolС
$sequential_3/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2&
$sequential_3/dropout_9/dropout/Constщ
"sequential_3/dropout_9/dropout/MulMul.sequential_3/max_pooling2d_11/MaxPool:output:0-sequential_3/dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2$
"sequential_3/dropout_9/dropout/Mulк
$sequential_3/dropout_9/dropout/ShapeShape.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_9/dropout/ShapeВ
;sequential_3/dropout_9/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02=
;sequential_3/dropout_9/dropout/random_uniform/RandomUniformг
-sequential_3/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2/
-sequential_3/dropout_9/dropout/GreaterEqual/yг
+sequential_3/dropout_9/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_9/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2-
+sequential_3/dropout_9/dropout/GreaterEqual═
#sequential_3/dropout_9/dropout/CastCast/sequential_3/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2%
#sequential_3/dropout_9/dropout/Cast▀
$sequential_3/dropout_9/dropout/Mul_1Mul&sequential_3/dropout_9/dropout/Mul:z:0'sequential_3/dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2&
$sequential_3/dropout_9/dropout/Mul_1Н
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_3/flatten_3/Const╨
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/dropout/Mul_1:z:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2 
sequential_3/flatten_3/Reshape╧
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp╘
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/MatMul╠
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp╓
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/BiasAddШ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/ReluУ
%sequential_3/dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_10/dropout/Const▌
#sequential_3/dropout_10/dropout/MulMul'sequential_3/dense_9/Relu:activations:0.sequential_3/dropout_10/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_3/dropout_10/dropout/Mulе
%sequential_3/dropout_10/dropout/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_10/dropout/Shape¤
<sequential_3/dropout_10/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_10/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_3/dropout_10/dropout/random_uniform/RandomUniformе
.sequential_3/dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_10/dropout/GreaterEqual/yЯ
,sequential_3/dropout_10/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_10/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_3/dropout_10/dropout/GreaterEqual╚
$sequential_3/dropout_10/dropout/CastCast0sequential_3/dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_3/dropout_10/dropout/Cast█
%sequential_3/dropout_10/dropout/Mul_1Mul'sequential_3/dropout_10/dropout/Mul:z:0(sequential_3/dropout_10/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_3/dropout_10/dropout/Mul_1╤
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp┘
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/dropout/Mul_1:z:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/MatMul╧
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp┌
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/BiasAddЫ
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/ReluУ
%sequential_3/dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_11/dropout/Const▐
#sequential_3/dropout_11/dropout/MulMul(sequential_3/dense_10/Relu:activations:0.sequential_3/dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_3/dropout_11/dropout/Mulж
%sequential_3/dropout_11/dropout/ShapeShape(sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_11/dropout/Shape¤
<sequential_3/dropout_11/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_3/dropout_11/dropout/random_uniform/RandomUniformе
.sequential_3/dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_11/dropout/GreaterEqual/yЯ
,sequential_3/dropout_11/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_11/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_3/dropout_11/dropout/GreaterEqual╚
$sequential_3/dropout_11/dropout/CastCast0sequential_3/dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_3/dropout_11/dropout/Cast█
%sequential_3/dropout_11/dropout/Mul_1Mul'sequential_3/dropout_11/dropout/Mul:z:0(sequential_3/dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_3/dropout_11/dropout/Mul_1й
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOp▒
dense_11/MatMulMatMul)sequential_3/dropout_11/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/MatMulз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpе
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_11/Softmaxу
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul█
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul▌
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul 
IdentityIdentitydense_11/Softmax:softmax:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp2^sequential_3/batch_normalization_3/AssignNewValue4^sequential_3/batch_normalization_3/AssignNewValue_1C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2f
1sequential_3/batch_normalization_3/AssignNewValue1sequential_3/batch_normalization_3/AssignNewValue2j
3sequential_3/batch_normalization_3/AssignNewValue_13sequential_3/batch_normalization_3/AssignNewValue_12И
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
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
:         KK
 
_user_specified_nameinputs
Э
А
E__inference_conv2d_10_layer_call_and_return_conditional_losses_572115

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
╬
б
*__inference_conv2d_10_layer_call_fn_572104

inputs"
unknown: А
	unknown_0:	А
identityИвStatefulPartitionedCallГ
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
GPU2 *0J 8В *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_5697312
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
▐
M
1__inference_max_pooling2d_11_layer_call_fn_569652

inputs
identityЄ
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
GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_5696462
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
┘
F
*__inference_flatten_3_layer_call_fn_572167

inputs
identity╩
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
GPU2 *0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_5697692
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
╕s
ы
__inference_call_489633

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
3sequential_3_dense_9_matmul_readvariableop_resource:АвАC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А6
(dense_11_biasadd_readvariableop_resource:
identityИвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpвBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_3/batch_normalization_3/ReadVariableOpв3sequential_3/batch_normalization_3/ReadVariableOp_1в-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpп
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack│
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1│
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2у
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice▌
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpу
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1Р
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1║
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3╫
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOpО
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D╬
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp╪
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2
sequential_3/conv2d_9/BiasAddЪ
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_3/conv2d_9/Reluц
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool█
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpИ
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D╥
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp▌
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2 
sequential_3/conv2d_10/BiasAddЮ
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_3/conv2d_10/Reluъ
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool▄
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpЙ
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D╥
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp▌
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_3/conv2d_11/BiasAddЮ
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_3/conv2d_11/Reluъ
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool▒
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*(
_output_shapes
:А		А2!
sequential_3/dropout_9/IdentityН
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_3/flatten_3/Const╚
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*!
_output_shapes
:ААв2 
sequential_3/flatten_3/Reshape╧
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp╠
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_9/MatMul╠
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp╬
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_9/BiasAddР
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_9/Reluд
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_3/dropout_10/Identity╤
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp╤
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_10/MatMul╧
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp╥
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_10/BiasAddУ
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_10/Reluе
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_3/dropout_11/Identityй
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOpй
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_11/MatMulз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpЭ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_11/BiasAddt
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_11/SoftmaxЄ
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2И
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
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
:АKK
 
_user_specified_nameinputs
Ц
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_569761

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
бu
Ф
H__inference_sequential_3_layer_call_and_return_conditional_losses_571602

inputs;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: C
(conv2d_10_conv2d_readvariableop_resource: А8
)conv2d_10_biasadd_readvariableop_resource:	АD
(conv2d_11_conv2d_readvariableop_resource:АА8
)conv2d_11_biasadd_readvariableop_resource:	А;
&dense_9_matmul_readvariableop_resource:АвА6
'dense_9_biasadd_readvariableop_resource:	А;
'dense_10_matmul_readvariableop_resource:
АА7
(dense_10_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1в conv2d_10/BiasAdd/ReadVariableOpвconv2d_10/Conv2D/ReadVariableOpв conv2d_11/BiasAdd/ReadVariableOpвconv2d_11/Conv2D/ReadVariableOpвconv2d_9/BiasAdd/ReadVariableOpвconv2d_9/Conv2D/ReadVariableOpв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpвdense_10/BiasAdd/ReadVariableOpвdense_10/MatMul/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpХ
lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_3/strided_slice/stackЩ
lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_3/strided_slice/stack_1Щ
lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_3/strided_slice/stack_2к
lambda_3/strided_sliceStridedSliceinputs%lambda_3/strided_slice/stack:output:0'lambda_3/strided_slice/stack_1:output:0'lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_3/strided_slice╢
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3lambda_3/strided_slice:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3░
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOpт
conv2d_9/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_9/Conv2Dз
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOpм
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_9/Relu╟
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool┤
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_10/Conv2D/ReadVariableOp▄
conv2d_10/Conv2DConv2D max_pooling2d_9/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_10/Conv2Dл
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp▒
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_10/Relu╦
max_pooling2d_10/MaxPoolMaxPoolconv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool╡
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_11/Conv2D/ReadVariableOp▌
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_11/Conv2Dл
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp▒
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_11/Relu╦
max_pooling2d_11/MaxPoolMaxPoolconv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPoolТ
dropout_9/IdentityIdentity!max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:         		А2
dropout_9/Identitys
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_3/ConstЬ
flatten_3/ReshapeReshapedropout_9/Identity:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_3/Reshapeи
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02
dense_9/MatMul/ReadVariableOpа
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_9/MatMulе
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_9/BiasAdd/ReadVariableOpв
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_9/ReluЕ
dropout_10/IdentityIdentitydense_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_10/Identityк
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_10/MatMul/ReadVariableOpе
dense_10/MatMulMatMuldropout_10/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_10/MatMulи
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_10/BiasAdd/ReadVariableOpж
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_10/ReluЖ
dropout_11/IdentityIdentitydense_10/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_11/Identity╓
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╬
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╨
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЯ
IdentityIdentitydropout_11/Identity:output:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
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
:         KK
 
_user_specified_nameinputs
╣[
В
H__inference_sequential_3_layer_call_and_return_conditional_losses_569850

inputs*
batch_normalization_3_569687:*
batch_normalization_3_569689:*
batch_normalization_3_569691:*
batch_normalization_3_569693:)
conv2d_9_569714: 
conv2d_9_569716: +
conv2d_10_569732: А
conv2d_10_569734:	А,
conv2d_11_569750:АА
conv2d_11_569752:	А#
dense_9_569789:АвА
dense_9_569791:	А#
dense_10_569819:
АА
dense_10_569821:	А
identityИв-batch_normalization_3/StatefulPartitionedCallв!conv2d_10/StatefulPartitionedCallв!conv2d_11/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpв dense_10/StatefulPartitionedCallв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_9/StatefulPartitionedCallв0dense_9/kernel/Regularizer/Square/ReadVariableOpс
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8В *M
fHRF
D__inference_lambda_3_layer_call_and_return_conditional_losses_5696672
lambda_3/PartitionedCall╜
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0batch_normalization_3_569687batch_normalization_3_569689batch_normalization_3_569691batch_normalization_3_569693*
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
GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5696862/
-batch_normalization_3/StatefulPartitionedCall╤
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_9_569714conv2d_9_569716*
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
GPU2 *0J 8В *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_5697132"
 conv2d_9/StatefulPartitionedCallЩ
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_5696222!
max_pooling2d_9/PartitionedCall╔
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0conv2d_10_569732conv2d_10_569734*
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
GPU2 *0J 8В *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_5697312#
!conv2d_10/StatefulPartitionedCallЮ
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_5696342"
 max_pooling2d_10/PartitionedCall╩
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_569750conv2d_11_569752*
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
GPU2 *0J 8В *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_5697492#
!conv2d_11/StatefulPartitionedCallЮ
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_5696462"
 max_pooling2d_11/PartitionedCallИ
dropout_9/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_5697612
dropout_9/PartitionedCall·
flatten_3/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
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
GPU2 *0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_5697692
flatten_3/PartitionedCall▒
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_569789dense_9_569791*
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_5697882!
dense_9/StatefulPartitionedCallВ
dropout_10/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_5697992
dropout_10/PartitionedCall╖
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_10_569819dense_10_569821*
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
GPU2 *0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5698182"
 dense_10/StatefulPartitionedCallГ
dropout_11/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5698292
dropout_11/PartitionedCall╛
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_569714*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╢
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_569789*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╕
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_569819* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulє
IdentityIdentity#dropout_11/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_9/StatefulPartitionedCall1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
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
:         KK
 
_user_specified_nameinputs
╦1
и
?__inference_CNN_layer_call_and_return_conditional_losses_570432

inputs!
sequential_3_570367:!
sequential_3_570369:!
sequential_3_570371:!
sequential_3_570373:-
sequential_3_570375: !
sequential_3_570377: .
sequential_3_570379: А"
sequential_3_570381:	А/
sequential_3_570383:АА"
sequential_3_570385:	А(
sequential_3_570387:АвА"
sequential_3_570389:	А'
sequential_3_570391:
АА"
sequential_3_570393:	А"
dense_11_570408:	А
dense_11_570410:
identityИв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpв dense_11/StatefulPartitionedCallв0dense_9/kernel/Regularizer/Square/ReadVariableOpв$sequential_3/StatefulPartitionedCall┬
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_570367sequential_3_570369sequential_3_570371sequential_3_570373sequential_3_570375sequential_3_570377sequential_3_570379sequential_3_570381sequential_3_570383sequential_3_570385sequential_3_570387sequential_3_570389sequential_3_570391sequential_3_570393*
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5698502&
$sequential_3/StatefulPartitionedCall└
 dense_11/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0dense_11_570408dense_11_570410*
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
GPU2 *0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5704072"
 dense_11/StatefulPartitionedCall┬
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_570375*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╗
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_570387*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╝
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_570391* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulт
IdentityIdentity)dense_11/StatefulPartitionedCall:output:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp!^dense_11/StatefulPartitionedCall1^dense_9/kernel/Regularizer/Square/ReadVariableOp%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ў
└
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_570039

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
╙
▓
__inference_loss_fn_1_572313N
9dense_9_kernel_regularizer_square_readvariableop_resource:АвА
identityИв0dense_9/kernel/Regularizer/Square/ReadVariableOpс
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_9_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulШ
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
Ш`
Ё
H__inference_sequential_3_layer_call_and_return_conditional_losses_570168

inputs*
batch_normalization_3_570108:*
batch_normalization_3_570110:*
batch_normalization_3_570112:*
batch_normalization_3_570114:)
conv2d_9_570117: 
conv2d_9_570119: +
conv2d_10_570123: А
conv2d_10_570125:	А,
conv2d_11_570129:АА
conv2d_11_570131:	А#
dense_9_570137:АвА
dense_9_570139:	А#
dense_10_570143:
АА
dense_10_570145:	А
identityИв-batch_normalization_3/StatefulPartitionedCallв!conv2d_10/StatefulPartitionedCallв!conv2d_11/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpв dense_10/StatefulPartitionedCallв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_9/StatefulPartitionedCallв0dense_9/kernel/Regularizer/Square/ReadVariableOpв"dropout_10/StatefulPartitionedCallв"dropout_11/StatefulPartitionedCallв!dropout_9/StatefulPartitionedCallс
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8В *M
fHRF
D__inference_lambda_3_layer_call_and_return_conditional_losses_5700662
lambda_3/PartitionedCall╗
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0batch_normalization_3_570108batch_normalization_3_570110batch_normalization_3_570112batch_normalization_3_570114*
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
GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5700392/
-batch_normalization_3/StatefulPartitionedCall╤
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_9_570117conv2d_9_570119*
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
GPU2 *0J 8В *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_5697132"
 conv2d_9/StatefulPartitionedCallЩ
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_5696222!
max_pooling2d_9/PartitionedCall╔
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0conv2d_10_570123conv2d_10_570125*
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
GPU2 *0J 8В *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_5697312#
!conv2d_10/StatefulPartitionedCallЮ
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_5696342"
 max_pooling2d_10/PartitionedCall╩
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_570129conv2d_11_570131*
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
GPU2 *0J 8В *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_5697492#
!conv2d_11/StatefulPartitionedCallЮ
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_5696462"
 max_pooling2d_11/PartitionedCallа
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_5699732#
!dropout_9/StatefulPartitionedCallВ
flatten_3/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_5697692
flatten_3/PartitionedCall▒
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_570137dense_9_570139*
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_5697882!
dense_9/StatefulPartitionedCall╛
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
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
GPU2 *0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_5699342$
"dropout_10/StatefulPartitionedCall┐
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_10_570143dense_10_570145*
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
GPU2 *0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5698182"
 dense_10/StatefulPartitionedCall└
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
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
GPU2 *0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5699012$
"dropout_11/StatefulPartitionedCall╛
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_570117*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╢
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_570137*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╕
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_570143* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulщ
IdentityIdentity+dropout_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_9/StatefulPartitionedCall1^dense_9/kernel/Regularizer/Square/ReadVariableOp#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
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
:         KK
 
_user_specified_nameinputs
Э
А
E__inference_conv2d_10_layer_call_and_return_conditional_losses_569731

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
Л
Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_569512

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
ж
╤
6__inference_batch_normalization_3_layer_call_fn_571978

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
:         KK*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5696862
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
╔
G
+__inference_dropout_10_layer_call_fn_572210

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
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_5697992
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
╕

Ў
D__inference_dense_11_layer_call_and_return_conditional_losses_571913

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
ШФ
п
?__inference_CNN_layer_call_and_return_conditional_losses_571258
input_1H
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
3sequential_3_dense_9_matmul_readvariableop_resource:АвАC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А6
(dense_11_biasadd_readvariableop_resource:
identityИв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpвBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_3/batch_normalization_3/ReadVariableOpв3sequential_3/batch_normalization_3/ReadVariableOp_1в-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpп
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack│
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1│
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2ь
#sequential_3/lambda_3/strided_sliceStridedSliceinput_12sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice▌
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpу
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1Р
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3╫
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOpЦ
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D╬
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpр
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/BiasAddв
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/Reluю
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool█
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpР
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D╥
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpх
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_3/conv2d_10/BiasAddж
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_3/conv2d_10/ReluЄ
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool▄
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpС
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D╥
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpх
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_3/conv2d_11/BiasAddж
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_3/conv2d_11/ReluЄ
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool╣
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:         		А2!
sequential_3/dropout_9/IdentityН
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_3/flatten_3/Const╨
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2 
sequential_3/flatten_3/Reshape╧
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp╘
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/MatMul╠
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp╓
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/BiasAddШ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/Reluм
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_3/dropout_10/Identity╤
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp┘
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/MatMul╧
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp┌
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/BiasAddЫ
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/Reluн
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_3/dropout_11/Identityй
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOp▒
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/MatMulз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpе
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_11/Softmaxу
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul█
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul▌
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulХ
IdentityIdentitydense_11/Softmax:softmax:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2И
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
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
:         KK
!
_user_specified_name	input_1
с
E
)__inference_lambda_3_layer_call_fn_571923

inputs
identity╧
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
GPU2 *0J 8В *M
fHRF
D__inference_lambda_3_layer_call_and_return_conditional_losses_5700662
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
ї
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_569973

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
м
h
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_569646

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
м▀
└!
"__inference__traced_restore_572675
file_prefix3
 assignvariableop_dense_11_kernel:	А.
 assignvariableop_1_dense_11_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_3_gamma:;
-assignvariableop_8_batch_normalization_3_beta:B
4assignvariableop_9_batch_normalization_3_moving_mean:G
9assignvariableop_10_batch_normalization_3_moving_variance:=
#assignvariableop_11_conv2d_9_kernel: /
!assignvariableop_12_conv2d_9_bias: ?
$assignvariableop_13_conv2d_10_kernel: А1
"assignvariableop_14_conv2d_10_bias:	А@
$assignvariableop_15_conv2d_11_kernel:АА1
"assignvariableop_16_conv2d_11_bias:	А7
"assignvariableop_17_dense_9_kernel:АвА/
 assignvariableop_18_dense_9_bias:	А7
#assignvariableop_19_dense_10_kernel:
АА0
!assignvariableop_20_dense_10_bias:	А#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_11_kernel_m:	А6
(assignvariableop_26_adam_dense_11_bias_m:D
6assignvariableop_27_adam_batch_normalization_3_gamma_m:C
5assignvariableop_28_adam_batch_normalization_3_beta_m:D
*assignvariableop_29_adam_conv2d_9_kernel_m: 6
(assignvariableop_30_adam_conv2d_9_bias_m: F
+assignvariableop_31_adam_conv2d_10_kernel_m: А8
)assignvariableop_32_adam_conv2d_10_bias_m:	АG
+assignvariableop_33_adam_conv2d_11_kernel_m:АА8
)assignvariableop_34_adam_conv2d_11_bias_m:	А>
)assignvariableop_35_adam_dense_9_kernel_m:АвА6
'assignvariableop_36_adam_dense_9_bias_m:	А>
*assignvariableop_37_adam_dense_10_kernel_m:
АА7
(assignvariableop_38_adam_dense_10_bias_m:	А=
*assignvariableop_39_adam_dense_11_kernel_v:	А6
(assignvariableop_40_adam_dense_11_bias_v:D
6assignvariableop_41_adam_batch_normalization_3_gamma_v:C
5assignvariableop_42_adam_batch_normalization_3_beta_v:D
*assignvariableop_43_adam_conv2d_9_kernel_v: 6
(assignvariableop_44_adam_conv2d_9_bias_v: F
+assignvariableop_45_adam_conv2d_10_kernel_v: А8
)assignvariableop_46_adam_conv2d_10_bias_v:	АG
+assignvariableop_47_adam_conv2d_11_kernel_v:АА8
)assignvariableop_48_adam_conv2d_11_bias_v:	А>
)assignvariableop_49_adam_dense_9_kernel_v:АвА6
'assignvariableop_50_adam_dense_9_bias_v:	А>
*assignvariableop_51_adam_dense_10_kernel_v:
АА7
(assignvariableop_52_adam_dense_10_bias_v:	А
identity_54ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ь
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*°
valueюBы6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names·
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╝
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ю
_output_shapes█
╪::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_3_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▓
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_3_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╣
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_3_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┴
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_3_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11л
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_9_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12й
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_9_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13м
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_10_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14к
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_10_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15м
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_11_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16к
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_11_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17к
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_9_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18и
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_9_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19л
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_10_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20й
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_10_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21б
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22б
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23г
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24г
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_11_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_11_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╛
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_3_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╜
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_3_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▓
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_9_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30░
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_9_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_10_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_10_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_11_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_11_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▒
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_9_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36п
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_9_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_10_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38░
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_10_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_11_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_11_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╛
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_3_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╜
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_3_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▓
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_9_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44░
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_9_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45│
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_10_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_10_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_11_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_11_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49▒
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_9_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50п
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_9_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51▓
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_10_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52░
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_10_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpь	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53▀	
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
╡
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_572291

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
л
g
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_569622

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
з
Щ
)__inference_dense_10_layer_call_fn_572247

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall·
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
GPU2 *0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5698182
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
╣
м
D__inference_dense_10_layer_call_and_return_conditional_losses_572264

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpП
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
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╒
d
+__inference_dropout_11_layer_call_fn_572274

inputs
identityИвStatefulPartitionedCallт
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
GPU2 *0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5699012
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
╣
м
D__inference_dense_10_layer_call_and_return_conditional_losses_569818

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpП
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
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Щ
н
$__inference_CNN_layer_call_fn_570893

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

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallп
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
:         *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_5704322
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

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
Ь
о
$__inference_CNN_layer_call_fn_570856
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

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCall░
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
:         *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_5704322
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
Л
Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572009

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
╡
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_569901

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
├
Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572045

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
╗Щ
ф
H__inference_sequential_3_layer_call_and_return_conditional_losses_571706

inputs;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: C
(conv2d_10_conv2d_readvariableop_resource: А8
)conv2d_10_biasadd_readvariableop_resource:	АD
(conv2d_11_conv2d_readvariableop_resource:АА8
)conv2d_11_biasadd_readvariableop_resource:	А;
&dense_9_matmul_readvariableop_resource:АвА6
'dense_9_biasadd_readvariableop_resource:	А;
'dense_10_matmul_readvariableop_resource:
АА7
(dense_10_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1в conv2d_10/BiasAdd/ReadVariableOpвconv2d_10/Conv2D/ReadVariableOpв conv2d_11/BiasAdd/ReadVariableOpвconv2d_11/Conv2D/ReadVariableOpвconv2d_9/BiasAdd/ReadVariableOpвconv2d_9/Conv2D/ReadVariableOpв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpвdense_10/BiasAdd/ReadVariableOpвdense_10/MatMul/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpХ
lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_3/strided_slice/stackЩ
lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_3/strided_slice/stack_1Щ
lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_3/strided_slice/stack_2к
lambda_3/strided_sliceStridedSliceinputs%lambda_3/strided_slice/stack:output:0'lambda_3/strided_slice/stack_1:output:0'lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_3/strided_slice╢
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ї
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3lambda_3/strided_slice:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_3/FusedBatchNormV3░
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue╝
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1░
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOpт
conv2d_9/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_9/Conv2Dз
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOpм
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_9/Relu╟
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool┤
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_10/Conv2D/ReadVariableOp▄
conv2d_10/Conv2DConv2D max_pooling2d_9/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_10/Conv2Dл
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp▒
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_10/Relu╦
max_pooling2d_10/MaxPoolMaxPoolconv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool╡
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_11/Conv2D/ReadVariableOp▌
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_11/Conv2Dл
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp▒
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_11/Relu╦
max_pooling2d_11/MaxPoolMaxPoolconv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
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
 *ф8О?2
dropout_9/dropout/Const╡
dropout_9/dropout/MulMul!max_pooling2d_11/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout_9/dropout/MulГ
dropout_9/dropout/ShapeShape!max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape█
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype020
.dropout_9/dropout/random_uniform/RandomUniformЙ
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2"
 dropout_9/dropout/GreaterEqual/yя
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2 
dropout_9/dropout/GreaterEqualж
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout_9/dropout/Castл
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout_9/dropout/Mul_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_3/ConstЬ
flatten_3/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_3/Reshapeи
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02
dense_9/MatMul/ReadVariableOpа
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_9/MatMulе
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_9/BiasAdd/ReadVariableOpв
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_9/Reluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_10/dropout/Constй
dropout_10/dropout/MulMuldense_9/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_10/dropout/Mul~
dropout_10/dropout/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape╓
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_10/dropout/random_uniform/RandomUniformЛ
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_10/dropout/GreaterEqual/yы
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_10/dropout/GreaterEqualб
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_10/dropout/Castз
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_10/dropout/Mul_1к
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_10/MatMul/ReadVariableOpе
dense_10/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_10/MatMulи
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_10/BiasAdd/ReadVariableOpж
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_10/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Constк
dropout_11/dropout/MulMuldense_10/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape╓
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_11/dropout/random_uniform/RandomUniformЛ
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/yы
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_11/dropout/GreaterEqualб
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_11/dropout/Castз
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_11/dropout/Mul_1╓
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╬
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╨
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulя
IdentityIdentitydropout_11/dropout/Mul_1:z:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
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
:         KK
 
_user_specified_nameinputs
Ц
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_572150

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
й
▒
D__inference_conv2d_9_layer_call_and_return_conditional_losses_569713

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpХ
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
Relu═
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╙
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╘
¤
-__inference_sequential_3_layer_call_fn_571453

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
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5698502
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ў
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_572279

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
ы
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_569769

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
Ъ
о
$__inference_CNN_layer_call_fn_570967
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

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallо
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
:         *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_5705722
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
├
Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_569686

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
ч
F
*__inference_dropout_9_layer_call_fn_572140

inputs
identity╤
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_5697612
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
▐
M
1__inference_max_pooling2d_10_layer_call_fn_569640

inputs
identityЄ
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
GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_5696342
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
В
╣
__inference_loss_fn_0_572302T
:conv2d_9_kernel_regularizer_square_readvariableop_resource: 
identityИв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpщ
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_9_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mulЪ
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
ы
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_572173

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
┬
`
D__inference_lambda_3_layer_call_and_return_conditional_losses_571931

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
┬
`
D__inference_lambda_3_layer_call_and_return_conditional_losses_569667

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
ў
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_569829

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
╔1
и
?__inference_CNN_layer_call_and_return_conditional_losses_570572

inputs!
sequential_3_570519:!
sequential_3_570521:!
sequential_3_570523:!
sequential_3_570525:-
sequential_3_570527: !
sequential_3_570529: .
sequential_3_570531: А"
sequential_3_570533:	А/
sequential_3_570535:АА"
sequential_3_570537:	А(
sequential_3_570539:АвА"
sequential_3_570541:	А'
sequential_3_570543:
АА"
sequential_3_570545:	А"
dense_11_570548:	А
dense_11_570550:
identityИв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpв dense_11/StatefulPartitionedCallв0dense_9/kernel/Regularizer/Square/ReadVariableOpв$sequential_3/StatefulPartitionedCall└
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_570519sequential_3_570521sequential_3_570523sequential_3_570525sequential_3_570527sequential_3_570529sequential_3_570531sequential_3_570533sequential_3_570535sequential_3_570537sequential_3_570539sequential_3_570541sequential_3_570543sequential_3_570545*
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5701682&
$sequential_3/StatefulPartitionedCall└
 dense_11/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0dense_11_570548dense_11_570550*
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
GPU2 *0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5704072"
 dense_11/StatefulPartitionedCall┬
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_570527*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╗
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_570539*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╝
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_570543* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulт
IdentityIdentity)dense_11/StatefulPartitionedCall:output:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp!^dense_11/StatefulPartitionedCall1^dense_9/kernel/Regularizer/Square/ReadVariableOp%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
є
c
*__inference_dropout_9_layer_call_fn_572145

inputs
identityИвStatefulPartitionedCallщ
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
GPU2 *0J 8В *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_5699732
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
°g
╗
__inference__traced_save_572506
file_prefix.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop$
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
ShardedFilenameц
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*°
valueюBы6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesъ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
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

identity_1Identity_1:output:0*ъ
_input_shapes╪
╒: :	А:: : : : : ::::: : : А:А:АА:А:АвА:А:
АА:А: : : : :	А:::: : : А:А:АА:А:АвА:А:
АА:А:	А:::: : : А:А:АА:А:АвА:А:
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
:А:
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
:А:'$#
!
_output_shapes
:АвА:!%

_output_shapes	
:А:&&"
 
_output_shapes
:
АА:!'
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
:А:6

_output_shapes
: 
ю
╤
6__inference_batch_normalization_3_layer_call_fn_571952

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
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5695122
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
╡
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_572232

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
й
л
C__inference_dense_9_layer_call_and_return_conditional_losses_569788

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpР
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
Relu╞
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╦
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
с
E
)__inference_lambda_3_layer_call_fn_571918

inputs
identity╧
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
GPU2 *0J 8В *M
fHRF
D__inference_lambda_3_layer_call_and_return_conditional_losses_5696672
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
╕s
ы
__inference_call_489705

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
3sequential_3_dense_9_matmul_readvariableop_resource:АвАC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А6
(dense_11_biasadd_readvariableop_resource:
identityИвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpвBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_3/batch_normalization_3/ReadVariableOpв3sequential_3/batch_normalization_3/ReadVariableOp_1в-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpп
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack│
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1│
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2у
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice▌
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpу
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1Р
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1║
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3╫
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOpО
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D╬
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp╪
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2
sequential_3/conv2d_9/BiasAddЪ
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_3/conv2d_9/Reluц
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool█
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpИ
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D╥
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp▌
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2 
sequential_3/conv2d_10/BiasAddЮ
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_3/conv2d_10/Reluъ
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool▄
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpЙ
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D╥
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp▌
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_3/conv2d_11/BiasAddЮ
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_3/conv2d_11/Reluъ
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool▒
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*(
_output_shapes
:А		А2!
sequential_3/dropout_9/IdentityН
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_3/flatten_3/Const╚
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*!
_output_shapes
:ААв2 
sequential_3/flatten_3/Reshape╧
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp╠
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_9/MatMul╠
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp╬
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_9/BiasAddР
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_9/Reluд
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_3/dropout_10/Identity╤
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp╤
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_10/MatMul╧
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp╥
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_10/BiasAddУ
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_3/dense_10/Reluе
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_3/dropout_11/Identityй
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOpй
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_11/MatMulз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpЭ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_11/BiasAddt
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_11/SoftmaxЄ
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2И
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
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
:АKK
 
_user_specified_nameinputs
▄
L
0__inference_max_pooling2d_9_layer_call_fn_569628

inputs
identityё
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
GPU2 *0J 8В *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_5696222
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
з
╝
!__inference__wrapped_model_569490
input_1

cnn_569456:

cnn_569458:

cnn_569460:

cnn_569462:$

cnn_569464: 

cnn_569466: %

cnn_569468: А

cnn_569470:	А&

cnn_569472:АА

cnn_569474:	А

cnn_569476:АвА

cnn_569478:	А

cnn_569480:
АА

cnn_569482:	А

cnn_569484:	А

cnn_569486:
identityИвCNN/StatefulPartitionedCallЭ
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1
cnn_569456
cnn_569458
cnn_569460
cnn_569462
cnn_569464
cnn_569466
cnn_569468
cnn_569470
cnn_569472
cnn_569474
cnn_569476
cnn_569478
cnn_569480
cnn_569482
cnn_569484
cnn_569486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В * 
fR
__inference_call_4876472
CNN/StatefulPartitionedCallЦ
IdentityIdentity$CNN/StatefulPartitionedCall:output:0^CNN/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2:
CNN/StatefulPartitionedCallCNN/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
м
h
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_569634

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
ъ
Е
-__inference_sequential_3_layer_call_fn_571519
lambda_3_input
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
identityИвStatefulPartitionedCallг
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
:         А*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5701682
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_3_input
┬
`
D__inference_lambda_3_layer_call_and_return_conditional_losses_570066

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
д
╤
6__inference_batch_normalization_3_layer_call_fn_571991

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallж
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
GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5700392
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
ш
│
__inference_loss_fn_2_572324N
:dense_10_kernel_regularizer_square_readvariableop_resource:
АА
identityИв1dense_10/kernel/Regularizer/Square/ReadVariableOpу
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_10_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЪ
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
г
Ч
)__inference_dense_11_layer_call_fn_571902

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall∙
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
GPU2 *0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5704072
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
╡
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_569934

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
й
л
C__inference_dense_9_layer_call_and_return_conditional_losses_572205

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpР
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
Relu╞
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╦
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
ХФ
о
?__inference_CNN_layer_call_and_return_conditional_losses_571057

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
3sequential_3_dense_9_matmul_readvariableop_resource:АвАC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А6
(dense_11_biasadd_readvariableop_resource:
identityИв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpвBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_3/batch_normalization_3/ReadVariableOpв3sequential_3/batch_normalization_3/ReadVariableOp_1в-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpп
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack│
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1│
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2ы
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice▌
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpу
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1Р
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3╫
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOpЦ
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D╬
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpр
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/BiasAddв
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/Reluю
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool█
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpР
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D╥
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpх
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_3/conv2d_10/BiasAddж
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_3/conv2d_10/ReluЄ
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool▄
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpС
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D╥
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpх
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_3/conv2d_11/BiasAddж
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_3/conv2d_11/ReluЄ
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool╣
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:         		А2!
sequential_3/dropout_9/IdentityН
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_3/flatten_3/Const╨
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2 
sequential_3/flatten_3/Reshape╧
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp╘
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/MatMul╠
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp╓
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/BiasAddШ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/Reluм
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_3/dropout_10/Identity╤
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp┘
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/MatMul╧
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp┌
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/BiasAddЫ
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/Reluн
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_3/dropout_11/Identityй
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOp▒
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/MatMulз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpе
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_11/Softmaxу
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul█
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul▌
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulХ
IdentityIdentitydense_11/Softmax:softmax:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2И
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
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
:         KK
 
_user_specified_nameinputs
╕

Ў
D__inference_dense_11_layer_call_and_return_conditional_losses_570407

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
ь
╤
6__inference_batch_normalization_3_layer_call_fn_571965

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall╕
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
GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5695562
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
б
Б
E__inference_conv2d_11_layer_call_and_return_conditional_losses_572135

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
┐
└
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572027

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
╒
d
+__inference_dropout_10_layer_call_fn_572215

inputs
identityИвStatefulPartitionedCallт
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
GPU2 *0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_5699342
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
╤
в
*__inference_conv2d_11_layer_call_fn_572124

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallГ
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
GPU2 *0J 8В *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_5697492
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
и
Щ
(__inference_dense_9_layer_call_fn_572188

inputs
unknown:АвА
	unknown_0:	А
identityИвStatefulPartitionedCall∙
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
GPU2 *0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_5697882
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
иu
ы
__inference_call_487647

inputsH
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
3sequential_3_dense_9_matmul_readvariableop_resource:АвАC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А6
(dense_11_biasadd_readvariableop_resource:
identityИвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpвBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_3/batch_normalization_3/ReadVariableOpв3sequential_3/batch_normalization_3/ReadVariableOp_1в-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpп
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack│
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1│
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2ы
#sequential_3/lambda_3/strided_sliceStridedSliceinputs2sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice▌
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpу
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1Р
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3╫
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOpЦ
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D╬
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpр
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/BiasAddв
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/Reluю
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool█
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpР
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D╥
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpх
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_3/conv2d_10/BiasAddж
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_3/conv2d_10/ReluЄ
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool▄
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpС
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D╥
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpх
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_3/conv2d_11/BiasAddж
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_3/conv2d_11/ReluЄ
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool╣
sequential_3/dropout_9/IdentityIdentity.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:         		А2!
sequential_3/dropout_9/IdentityН
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_3/flatten_3/Const╨
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2 
sequential_3/flatten_3/Reshape╧
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp╘
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/MatMul╠
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp╓
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/BiasAddШ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/Reluм
 sequential_3/dropout_10/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_3/dropout_10/Identity╤
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp┘
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/MatMul╧
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp┌
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/BiasAddЫ
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/Reluн
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_3/dropout_11/Identityй
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOp▒
dense_11/MatMulMatMul)sequential_3/dropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/MatMulз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpе
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_11/Softmax·
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOpC^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2И
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
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
:         KK
 
_user_specified_nameinputs
Ч
н
$__inference_CNN_layer_call_fn_570930

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

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallн
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
:         *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_5705722
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

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
ь
Е
-__inference_sequential_3_layer_call_fn_571420
lambda_3_input
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
identityИвStatefulPartitionedCallе
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
:         А*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5698502
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_3_input
ў
└
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572063

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
й
▒
D__inference_conv2d_9_layer_call_and_return_conditional_losses_572095

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpХ
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
Relu═
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╙
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ў
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_569799

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
ў
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_572220

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
┬
`
D__inference_lambda_3_layer_call_and_return_conditional_losses_571939

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
ї
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_572162

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
╣u
Ь
H__inference_sequential_3_layer_call_and_return_conditional_losses_571789
lambda_3_input;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: C
(conv2d_10_conv2d_readvariableop_resource: А8
)conv2d_10_biasadd_readvariableop_resource:	АD
(conv2d_11_conv2d_readvariableop_resource:АА8
)conv2d_11_biasadd_readvariableop_resource:	А;
&dense_9_matmul_readvariableop_resource:АвА6
'dense_9_biasadd_readvariableop_resource:	А;
'dense_10_matmul_readvariableop_resource:
АА7
(dense_10_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1в conv2d_10/BiasAdd/ReadVariableOpвconv2d_10/Conv2D/ReadVariableOpв conv2d_11/BiasAdd/ReadVariableOpвconv2d_11/Conv2D/ReadVariableOpвconv2d_9/BiasAdd/ReadVariableOpвconv2d_9/Conv2D/ReadVariableOpв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpвdense_10/BiasAdd/ReadVariableOpвdense_10/MatMul/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpХ
lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_3/strided_slice/stackЩ
lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_3/strided_slice/stack_1Щ
lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_3/strided_slice/stack_2▓
lambda_3/strided_sliceStridedSlicelambda_3_input%lambda_3/strided_slice/stack:output:0'lambda_3/strided_slice/stack_1:output:0'lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_3/strided_slice╢
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3lambda_3/strided_slice:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3░
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOpт
conv2d_9/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_9/Conv2Dз
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOpм
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_9/Relu╟
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool┤
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_10/Conv2D/ReadVariableOp▄
conv2d_10/Conv2DConv2D max_pooling2d_9/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_10/Conv2Dл
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp▒
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_10/Relu╦
max_pooling2d_10/MaxPoolMaxPoolconv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool╡
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_11/Conv2D/ReadVariableOp▌
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_11/Conv2Dл
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp▒
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_11/Relu╦
max_pooling2d_11/MaxPoolMaxPoolconv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPoolТ
dropout_9/IdentityIdentity!max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:         		А2
dropout_9/Identitys
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_3/ConstЬ
flatten_3/ReshapeReshapedropout_9/Identity:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_3/Reshapeи
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02
dense_9/MatMul/ReadVariableOpа
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_9/MatMulе
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_9/BiasAdd/ReadVariableOpв
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_9/ReluЕ
dropout_10/IdentityIdentitydense_9/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_10/Identityк
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_10/MatMul/ReadVariableOpе
dense_10/MatMulMatMuldropout_10/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_10/MatMulи
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_10/BiasAdd/ReadVariableOpж
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_10/ReluЖ
dropout_11/IdentityIdentitydense_10/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_11/Identity╓
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╬
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╨
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЯ
IdentityIdentitydropout_11/Identity:output:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
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
:         KK
(
_user_specified_namelambda_3_input
╙Щ
ь
H__inference_sequential_3_layer_call_and_return_conditional_losses_571893
lambda_3_input;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: C
(conv2d_10_conv2d_readvariableop_resource: А8
)conv2d_10_biasadd_readvariableop_resource:	АD
(conv2d_11_conv2d_readvariableop_resource:АА8
)conv2d_11_biasadd_readvariableop_resource:	А;
&dense_9_matmul_readvariableop_resource:АвА6
'dense_9_biasadd_readvariableop_resource:	А;
'dense_10_matmul_readvariableop_resource:
АА7
(dense_10_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1в conv2d_10/BiasAdd/ReadVariableOpвconv2d_10/Conv2D/ReadVariableOpв conv2d_11/BiasAdd/ReadVariableOpвconv2d_11/Conv2D/ReadVariableOpвconv2d_9/BiasAdd/ReadVariableOpвconv2d_9/Conv2D/ReadVariableOpв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpвdense_10/BiasAdd/ReadVariableOpвdense_10/MatMul/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpХ
lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_3/strided_slice/stackЩ
lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_3/strided_slice/stack_1Щ
lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_3/strided_slice/stack_2▓
lambda_3/strided_sliceStridedSlicelambda_3_input%lambda_3/strided_slice/stack:output:0'lambda_3/strided_slice/stack_1:output:0'lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_3/strided_slice╢
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ї
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3lambda_3/strided_slice:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_3/FusedBatchNormV3░
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue╝
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1░
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOpт
conv2d_9/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_9/Conv2Dз
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOpм
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_9/Relu╟
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool┤
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_10/Conv2D/ReadVariableOp▄
conv2d_10/Conv2DConv2D max_pooling2d_9/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_10/Conv2Dл
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp▒
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_10/Relu╦
max_pooling2d_10/MaxPoolMaxPoolconv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool╡
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_11/Conv2D/ReadVariableOp▌
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_11/Conv2Dл
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp▒
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_11/Relu╦
max_pooling2d_11/MaxPoolMaxPoolconv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
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
 *ф8О?2
dropout_9/dropout/Const╡
dropout_9/dropout/MulMul!max_pooling2d_11/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2
dropout_9/dropout/MulГ
dropout_9/dropout/ShapeShape!max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape█
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype020
.dropout_9/dropout/random_uniform/RandomUniformЙ
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2"
 dropout_9/dropout/GreaterEqual/yя
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2 
dropout_9/dropout/GreaterEqualж
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2
dropout_9/dropout/Castл
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2
dropout_9/dropout/Mul_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_3/ConstЬ
flatten_3/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2
flatten_3/Reshapeи
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02
dense_9/MatMul/ReadVariableOpа
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_9/MatMulе
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_9/BiasAdd/ReadVariableOpв
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_9/Reluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_10/dropout/Constй
dropout_10/dropout/MulMuldense_9/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_10/dropout/Mul~
dropout_10/dropout/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape╓
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_10/dropout/random_uniform/RandomUniformЛ
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_10/dropout/GreaterEqual/yы
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_10/dropout/GreaterEqualб
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_10/dropout/Castз
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_10/dropout/Mul_1к
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_10/MatMul/ReadVariableOpе
dense_10/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_10/MatMulи
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_10/BiasAdd/ReadVariableOpж
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_10/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Constк
dropout_11/dropout/MulMuldense_10/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape╓
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_11/dropout/random_uniform/RandomUniformЛ
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/yы
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_11/dropout/GreaterEqualб
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_11/dropout/Castз
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_11/dropout/Mul_1╓
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul╬
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul╨
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulя
IdentityIdentitydropout_11/dropout/Mul_1:z:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
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
:         KK
(
_user_specified_namelambda_3_input
б
Б
E__inference_conv2d_11_layer_call_and_return_conditional_losses_569749

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
╔
G
+__inference_dropout_11_layer_call_fn_572269

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
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5698292
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
╥
¤
-__inference_sequential_3_layer_call_fn_571486

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
identityИвStatefulPartitionedCallЫ
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
GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5701682
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
√└
Щ
?__inference_CNN_layer_call_and_return_conditional_losses_571369
input_1H
:sequential_3_batch_normalization_3_readvariableop_resource:J
<sequential_3_batch_normalization_3_readvariableop_1_resource:Y
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_3_conv2d_9_conv2d_readvariableop_resource: C
5sequential_3_conv2d_9_biasadd_readvariableop_resource: P
5sequential_3_conv2d_10_conv2d_readvariableop_resource: АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
3sequential_3_dense_9_matmul_readvariableop_resource:АвАC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А6
(dense_11_biasadd_readvariableop_resource:
identityИв1conv2d_9/kernel/Regularizer/Square/ReadVariableOpв1dense_10/kernel/Regularizer/Square/ReadVariableOpвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpв0dense_9/kernel/Regularizer/Square/ReadVariableOpв1sequential_3/batch_normalization_3/AssignNewValueв3sequential_3/batch_normalization_3/AssignNewValue_1вBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_3/batch_normalization_3/ReadVariableOpв3sequential_3/batch_normalization_3/ReadVariableOp_1в-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpп
)sequential_3/lambda_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_3/lambda_3/strided_slice/stack│
+sequential_3/lambda_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_3/lambda_3/strided_slice/stack_1│
+sequential_3/lambda_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_3/lambda_3/strided_slice/stack_2ь
#sequential_3/lambda_3/strided_sliceStridedSliceinput_12sequential_3/lambda_3/strided_slice/stack:output:04sequential_3/lambda_3/strided_slice/stack_1:output:04sequential_3/lambda_3/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_3/lambda_3/strided_slice▌
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpу
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1Р
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1╨
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,sequential_3/lambda_3/strided_slice:output:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<25
3sequential_3/batch_normalization_3/FusedBatchNormV3ё
1sequential_3/batch_normalization_3/AssignNewValueAssignVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource@sequential_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_3/batch_normalization_3/AssignNewValue¤
3sequential_3/batch_normalization_3/AssignNewValue_1AssignVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceDsequential_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0E^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_3/batch_normalization_3/AssignNewValue_1╫
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_9/Conv2D/ReadVariableOpЦ
sequential_3/conv2d_9/Conv2DConv2D7sequential_3/batch_normalization_3/FusedBatchNormV3:y:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_3/conv2d_9/Conv2D╬
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpр
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/BiasAddв
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_3/conv2d_9/Reluю
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_9/MaxPool█
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpР
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_3/conv2d_10/Conv2D╥
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpх
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_3/conv2d_10/BiasAddж
sequential_3/conv2d_10/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_3/conv2d_10/ReluЄ
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool▄
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpС
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_3/conv2d_11/Conv2D╥
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpх
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_3/conv2d_11/BiasAddж
sequential_3/conv2d_11/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_3/conv2d_11/ReluЄ
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPoolС
$sequential_3/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2&
$sequential_3/dropout_9/dropout/Constщ
"sequential_3/dropout_9/dropout/MulMul.sequential_3/max_pooling2d_11/MaxPool:output:0-sequential_3/dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:         		А2$
"sequential_3/dropout_9/dropout/Mulк
$sequential_3/dropout_9/dropout/ShapeShape.sequential_3/max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_9/dropout/ShapeВ
;sequential_3/dropout_9/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype02=
;sequential_3/dropout_9/dropout/random_uniform/RandomUniformг
-sequential_3/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2/
-sequential_3/dropout_9/dropout/GreaterEqual/yг
+sequential_3/dropout_9/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_9/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А2-
+sequential_3/dropout_9/dropout/GreaterEqual═
#sequential_3/dropout_9/dropout/CastCast/sequential_3/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		А2%
#sequential_3/dropout_9/dropout/Cast▀
$sequential_3/dropout_9/dropout/Mul_1Mul&sequential_3/dropout_9/dropout/Mul:z:0'sequential_3/dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:         		А2&
$sequential_3/dropout_9/dropout/Mul_1Н
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_3/flatten_3/Const╨
sequential_3/flatten_3/ReshapeReshape(sequential_3/dropout_9/dropout/Mul_1:z:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:         Ав2 
sequential_3/flatten_3/Reshape╧
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype02,
*sequential_3/dense_9/MatMul/ReadVariableOp╘
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/MatMul╠
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp╓
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/BiasAddШ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_9/ReluУ
%sequential_3/dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_10/dropout/Const▌
#sequential_3/dropout_10/dropout/MulMul'sequential_3/dense_9/Relu:activations:0.sequential_3/dropout_10/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_3/dropout_10/dropout/Mulе
%sequential_3/dropout_10/dropout/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_10/dropout/Shape¤
<sequential_3/dropout_10/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_10/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_3/dropout_10/dropout/random_uniform/RandomUniformе
.sequential_3/dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_10/dropout/GreaterEqual/yЯ
,sequential_3/dropout_10/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_10/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_10/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_3/dropout_10/dropout/GreaterEqual╚
$sequential_3/dropout_10/dropout/CastCast0sequential_3/dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_3/dropout_10/dropout/Cast█
%sequential_3/dropout_10/dropout/Mul_1Mul'sequential_3/dropout_10/dropout/Mul:z:0(sequential_3/dropout_10/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_3/dropout_10/dropout/Mul_1╤
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_3/dense_10/MatMul/ReadVariableOp┘
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_10/dropout/Mul_1:z:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/MatMul╧
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp┌
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/BiasAddЫ
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_3/dense_10/ReluУ
%sequential_3/dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_3/dropout_11/dropout/Const▐
#sequential_3/dropout_11/dropout/MulMul(sequential_3/dense_10/Relu:activations:0.sequential_3/dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_3/dropout_11/dropout/Mulж
%sequential_3/dropout_11/dropout/ShapeShape(sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dropout_11/dropout/Shape¤
<sequential_3/dropout_11/dropout/random_uniform/RandomUniformRandomUniform.sequential_3/dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_3/dropout_11/dropout/random_uniform/RandomUniformе
.sequential_3/dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_3/dropout_11/dropout/GreaterEqual/yЯ
,sequential_3/dropout_11/dropout/GreaterEqualGreaterEqualEsequential_3/dropout_11/dropout/random_uniform/RandomUniform:output:07sequential_3/dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_3/dropout_11/dropout/GreaterEqual╚
$sequential_3/dropout_11/dropout/CastCast0sequential_3/dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_3/dropout_11/dropout/Cast█
%sequential_3/dropout_11/dropout/Mul_1Mul'sequential_3/dropout_11/dropout/Mul:z:0(sequential_3/dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_3/dropout_11/dropout/Mul_1й
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOp▒
dense_11/MatMulMatMul)sequential_3/dropout_11/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/MatMulз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpе
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_11/Softmaxу
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp╛
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_9/kernel/Regularizer/SquareЯ
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/Const╛
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/SumЛ
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!conv2d_9/kernel/Regularizer/mul/x└
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul█
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOp╢
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:АвА2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/Const║
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2"
 dense_9/kernel/Regularizer/mul/x╝
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mul▌
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOp╕
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/Const╛
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_10/kernel/Regularizer/mul/x└
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mul 
IdentityIdentitydense_11/Softmax:softmax:02^conv2d_9/kernel/Regularizer/Square/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/Square/ReadVariableOp2^sequential_3/batch_normalization_3/AssignNewValue4^sequential_3/batch_normalization_3/AssignNewValue_1C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_3/ReadVariableOp4^sequential_3/batch_normalization_3/ReadVariableOp_1.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/Square/ReadVariableOp0dense_9/kernel/Regularizer/Square/ReadVariableOp2f
1sequential_3/batch_normalization_3/AssignNewValue1sequential_3/batch_normalization_3/AssignNewValue2j
3sequential_3/batch_normalization_3/AssignNewValue_13sequential_3/batch_normalization_3/AssignNewValue_12И
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
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
:         KK
!
_user_specified_name	input_1
┐
└
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_569556

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
StatefulPartitionedCall:0         tensorflow/serving/predict:─Г
■


h2ptjl
_output
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
щ__call__
+ъ&call_and_return_all_conditional_losses
ы_default_save_signature
	ьcall"М	
_tf_keras_modelЄ{"name": "CNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
╕h
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
trainable_variables
regularization_losses
	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"╤d
_tf_keras_sequential▓d{"name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_3_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_3_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_3_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
╫

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
я__call__
+Ё&call_and_return_all_conditional_losses"░
_tf_keras_layerЦ{"name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
ы
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m╧'m╨*m╤+m╥,m╙-m╘.m╒/m╓0m╫1m╪2m┘3m┌v█v▄&v▌'v▐*v▀+vр,vс-vт.vу/vф0vх1vц2vч3vш"
	optimizer
Ц
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
Ж
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
 "
trackable_list_wrapper
╬
4layer_metrics
	variables
5layer_regularization_losses
6metrics
7non_trainable_variables
trainable_variables

8layers
regularization_losses
щ__call__
ы_default_save_signature
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
-
ёserving_default"
signature_map
╪
9	variables
:regularization_losses
;trainable_variables
<	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"╟
_tf_keras_layerн{"name": "lambda_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_3", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
─

=axis
	&gamma
'beta
(moving_mean
)moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"ю
_tf_keras_layer╘{"name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
а

*kernel
+bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses"∙	
_tf_keras_layer▀	{"name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
▒
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
°__call__
+∙&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"name": "max_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
╓


,kernel
-bias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
·__call__
+√&call_and_return_all_conditional_losses"п	
_tf_keras_layerХ	{"name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
│
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
№__call__
+¤&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"name": "max_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
╪


.kernel
/bias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
■__call__
+ &call_and_return_all_conditional_losses"▒	
_tf_keras_layerЧ	{"name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
│
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"name": "max_pooling2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
 
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"ю
_tf_keras_layer╘{"name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
Ш
^	variables
_regularization_losses
`trainable_variables
a	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"З
_tf_keras_layerэ{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
и	

0kernel
1bias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"Б
_tf_keras_layerч{"name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
Б
f	variables
gregularization_losses
htrainable_variables
i	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
ж	

2kernel
3bias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
К__call__
+Л&call_and_return_all_conditional_losses" 
_tf_keras_layerх{"name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Б
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
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
8
О0
П1
Р2"
trackable_list_wrapper
░
rlayer_metrics
	variables
slayer_regularization_losses
tmetrics
unon_trainable_variables
trainable_variables

vlayers
regularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_11/kernel
:2dense_11/bias
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
░
wlayer_metrics
	variables
xmetrics
ynon_trainable_variables
regularization_losses
trainable_variables

zlayers
{layer_regularization_losses
я__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
):' 2conv2d_9/kernel
: 2conv2d_9/bias
+:) А2conv2d_10/kernel
:А2conv2d_10/bias
,:*АА2conv2d_11/kernel
:А2conv2d_11/bias
#:!АвА2dense_9/kernel
:А2dense_9/bias
#:!
АА2dense_10/kernel
:А2dense_10/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
(0
)1"
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
│
~layer_metrics
9	variables
metrics
Аnon_trainable_variables
:regularization_losses
;trainable_variables
Бlayers
 Вlayer_regularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
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
╡
Гlayer_metrics
>	variables
Дmetrics
Еnon_trainable_variables
?regularization_losses
@trainable_variables
Жlayers
 Зlayer_regularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
(
О0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
╡
Иlayer_metrics
B	variables
Йmetrics
Кnon_trainable_variables
Cregularization_losses
Dtrainable_variables
Лlayers
 Мlayer_regularization_losses
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Нlayer_metrics
F	variables
Оmetrics
Пnon_trainable_variables
Gregularization_losses
Htrainable_variables
Рlayers
 Сlayer_regularization_losses
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
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
Тlayer_metrics
J	variables
Уmetrics
Фnon_trainable_variables
Kregularization_losses
Ltrainable_variables
Хlayers
 Цlayer_regularization_losses
·__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Чlayer_metrics
N	variables
Шmetrics
Щnon_trainable_variables
Oregularization_losses
Ptrainable_variables
Ъlayers
 Ыlayer_regularization_losses
№__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
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
Ьlayer_metrics
R	variables
Эmetrics
Юnon_trainable_variables
Sregularization_losses
Ttrainable_variables
Яlayers
 аlayer_regularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
бlayer_metrics
V	variables
вmetrics
гnon_trainable_variables
Wregularization_losses
Xtrainable_variables
дlayers
 еlayer_regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
жlayer_metrics
Z	variables
зmetrics
иnon_trainable_variables
[regularization_losses
\trainable_variables
йlayers
 кlayer_regularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
лlayer_metrics
^	variables
мmetrics
нnon_trainable_variables
_regularization_losses
`trainable_variables
оlayers
 пlayer_regularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
(
П0"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
╡
░layer_metrics
b	variables
▒metrics
▓non_trainable_variables
cregularization_losses
dtrainable_variables
│layers
 ┤layer_regularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╡layer_metrics
f	variables
╢metrics
╖non_trainable_variables
gregularization_losses
htrainable_variables
╕layers
 ╣layer_regularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
(
Р0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
╡
║layer_metrics
j	variables
╗metrics
╝non_trainable_variables
kregularization_losses
ltrainable_variables
╜layers
 ╛layer_regularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
┐layer_metrics
n	variables
└metrics
┴non_trainable_variables
oregularization_losses
ptrainable_variables
┬layers
 ├layer_regularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╪

─total

┼count
╞	variables
╟	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
Ы

╚total

╔count
╩
_fn_kwargs
╦	variables
╠	keras_api"╧
_tf_keras_metric┤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
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
 "
trackable_list_wrapper
(
Р0"
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
:  (2total
:  (2count
0
─0
┼1"
trackable_list_wrapper
.
╞	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╚0
╔1"
trackable_list_wrapper
.
╦	variables"
_generic_user_object
':%	А2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
.:,2"Adam/batch_normalization_3/gamma/m
-:+2!Adam/batch_normalization_3/beta/m
.:, 2Adam/conv2d_9/kernel/m
 : 2Adam/conv2d_9/bias/m
0:. А2Adam/conv2d_10/kernel/m
": А2Adam/conv2d_10/bias/m
1:/АА2Adam/conv2d_11/kernel/m
": А2Adam/conv2d_11/bias/m
(:&АвА2Adam/dense_9/kernel/m
 :А2Adam/dense_9/bias/m
(:&
АА2Adam/dense_10/kernel/m
!:А2Adam/dense_10/bias/m
':%	А2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
.:,2"Adam/batch_normalization_3/gamma/v
-:+2!Adam/batch_normalization_3/beta/v
.:, 2Adam/conv2d_9/kernel/v
 : 2Adam/conv2d_9/bias/v
0:. А2Adam/conv2d_10/kernel/v
": А2Adam/conv2d_10/bias/v
1:/АА2Adam/conv2d_11/kernel/v
": А2Adam/conv2d_11/bias/v
(:&АвА2Adam/dense_9/kernel/v
 :А2Adam/dense_9/bias/v
(:&
АА2Adam/dense_10/kernel/v
!:А2Adam/dense_10/bias/v
╥2╧
$__inference_CNN_layer_call_fn_570856
$__inference_CNN_layer_call_fn_570893
$__inference_CNN_layer_call_fn_570930
$__inference_CNN_layer_call_fn_570967┤
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
╛2╗
?__inference_CNN_layer_call_and_return_conditional_losses_571057
?__inference_CNN_layer_call_and_return_conditional_losses_571168
?__inference_CNN_layer_call_and_return_conditional_losses_571258
?__inference_CNN_layer_call_and_return_conditional_losses_571369┤
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
ч2ф
!__inference__wrapped_model_569490╛
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
Д2Б
__inference_call_489633
__inference_call_489705
__inference_call_489777│
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
В2 
-__inference_sequential_3_layer_call_fn_571420
-__inference_sequential_3_layer_call_fn_571453
-__inference_sequential_3_layer_call_fn_571486
-__inference_sequential_3_layer_call_fn_571519└
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
ю2ы
H__inference_sequential_3_layer_call_and_return_conditional_losses_571602
H__inference_sequential_3_layer_call_and_return_conditional_losses_571706
H__inference_sequential_3_layer_call_and_return_conditional_losses_571789
H__inference_sequential_3_layer_call_and_return_conditional_losses_571893└
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
╙2╨
)__inference_dense_11_layer_call_fn_571902в
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
ю2ы
D__inference_dense_11_layer_call_and_return_conditional_losses_571913в
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
╦B╚
$__inference_signature_wrapper_570819input_1"Ф
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
Ь2Щ
)__inference_lambda_3_layer_call_fn_571918
)__inference_lambda_3_layer_call_fn_571923└
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
╥2╧
D__inference_lambda_3_layer_call_and_return_conditional_losses_571931
D__inference_lambda_3_layer_call_and_return_conditional_losses_571939└
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
Ъ2Ч
6__inference_batch_normalization_3_layer_call_fn_571952
6__inference_batch_normalization_3_layer_call_fn_571965
6__inference_batch_normalization_3_layer_call_fn_571978
6__inference_batch_normalization_3_layer_call_fn_571991┤
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
Ж2Г
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572009
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572027
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572045
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572063┤
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
╙2╨
)__inference_conv2d_9_layer_call_fn_572078в
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
ю2ы
D__inference_conv2d_9_layer_call_and_return_conditional_losses_572095в
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
Ш2Х
0__inference_max_pooling2d_9_layer_call_fn_569628р
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
│2░
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_569622р
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
╘2╤
*__inference_conv2d_10_layer_call_fn_572104в
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
E__inference_conv2d_10_layer_call_and_return_conditional_losses_572115в
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
Щ2Ц
1__inference_max_pooling2d_10_layer_call_fn_569640р
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
┤2▒
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_569634р
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
╘2╤
*__inference_conv2d_11_layer_call_fn_572124в
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
E__inference_conv2d_11_layer_call_and_return_conditional_losses_572135в
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
Щ2Ц
1__inference_max_pooling2d_11_layer_call_fn_569652р
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
┤2▒
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_569646р
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
Т2П
*__inference_dropout_9_layer_call_fn_572140
*__inference_dropout_9_layer_call_fn_572145┤
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
╚2┼
E__inference_dropout_9_layer_call_and_return_conditional_losses_572150
E__inference_dropout_9_layer_call_and_return_conditional_losses_572162┤
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
*__inference_flatten_3_layer_call_fn_572167в
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_572173в
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
╥2╧
(__inference_dense_9_layer_call_fn_572188в
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
э2ъ
C__inference_dense_9_layer_call_and_return_conditional_losses_572205в
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
Ф2С
+__inference_dropout_10_layer_call_fn_572210
+__inference_dropout_10_layer_call_fn_572215┤
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
╩2╟
F__inference_dropout_10_layer_call_and_return_conditional_losses_572220
F__inference_dropout_10_layer_call_and_return_conditional_losses_572232┤
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
╙2╨
)__inference_dense_10_layer_call_fn_572247в
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
ю2ы
D__inference_dense_10_layer_call_and_return_conditional_losses_572264в
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
Ф2С
+__inference_dropout_11_layer_call_fn_572269
+__inference_dropout_11_layer_call_fn_572274┤
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
╩2╟
F__inference_dropout_11_layer_call_and_return_conditional_losses_572279
F__inference_dropout_11_layer_call_and_return_conditional_losses_572291┤
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
│2░
__inference_loss_fn_0_572302П
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
│2░
__inference_loss_fn_1_572313П
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
│2░
__inference_loss_fn_2_572324П
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
annotationsк *в ╣
?__inference_CNN_layer_call_and_return_conditional_losses_571057v&'()*+,-./0123;в8
1в.
(К%
inputs         KK
p 
к "%в"
К
0         
Ъ ╣
?__inference_CNN_layer_call_and_return_conditional_losses_571168v&'()*+,-./0123;в8
1в.
(К%
inputs         KK
p
к "%в"
К
0         
Ъ ║
?__inference_CNN_layer_call_and_return_conditional_losses_571258w&'()*+,-./0123<в9
2в/
)К&
input_1         KK
p 
к "%в"
К
0         
Ъ ║
?__inference_CNN_layer_call_and_return_conditional_losses_571369w&'()*+,-./0123<в9
2в/
)К&
input_1         KK
p
к "%в"
К
0         
Ъ Т
$__inference_CNN_layer_call_fn_570856j&'()*+,-./0123<в9
2в/
)К&
input_1         KK
p 
к "К         С
$__inference_CNN_layer_call_fn_570893i&'()*+,-./0123;в8
1в.
(К%
inputs         KK
p 
к "К         С
$__inference_CNN_layer_call_fn_570930i&'()*+,-./0123;в8
1в.
(К%
inputs         KK
p
к "К         Т
$__inference_CNN_layer_call_fn_570967j&'()*+,-./0123<в9
2в/
)К&
input_1         KK
p
к "К         з
!__inference__wrapped_model_569490Б&'()*+,-./01238в5
.в+
)К&
input_1         KK
к "3к0
.
output_1"К
output_1         ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572009Ц&'()MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572027Ц&'()MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╟
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572045r&'();в8
1в.
(К%
inputs         KK
p 
к "-в*
#К 
0         KK
Ъ ╟
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_572063r&'();в8
1в.
(К%
inputs         KK
p
к "-в*
#К 
0         KK
Ъ ─
6__inference_batch_normalization_3_layer_call_fn_571952Й&'()MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ─
6__inference_batch_normalization_3_layer_call_fn_571965Й&'()MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           Я
6__inference_batch_normalization_3_layer_call_fn_571978e&'();в8
1в.
(К%
inputs         KK
p 
к " К         KKЯ
6__inference_batch_normalization_3_layer_call_fn_571991e&'();в8
1в.
(К%
inputs         KK
p
к " К         KKt
__inference_call_489633Y&'()*+,-./01233в0
)в&
 К
inputsАKK
p
к "К	Аt
__inference_call_489705Y&'()*+,-./01233в0
)в&
 К
inputsАKK
p 
к "К	АД
__inference_call_489777i&'()*+,-./0123;в8
1в.
(К%
inputs         KK
p 
к "К         ╢
E__inference_conv2d_10_layer_call_and_return_conditional_losses_572115m,-7в4
-в*
(К%
inputs         %% 
к ".в+
$К!
0         %%А
Ъ О
*__inference_conv2d_10_layer_call_fn_572104`,-7в4
-в*
(К%
inputs         %% 
к "!К         %%А╖
E__inference_conv2d_11_layer_call_and_return_conditional_losses_572135n./8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ П
*__inference_conv2d_11_layer_call_fn_572124a./8в5
.в+
)К&
inputs         А
к "!К         А┤
D__inference_conv2d_9_layer_call_and_return_conditional_losses_572095l*+7в4
-в*
(К%
inputs         KK
к "-в*
#К 
0         KK 
Ъ М
)__inference_conv2d_9_layer_call_fn_572078_*+7в4
-в*
(К%
inputs         KK
к " К         KK ж
D__inference_dense_10_layer_call_and_return_conditional_losses_572264^230в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ ~
)__inference_dense_10_layer_call_fn_572247Q230в-
&в#
!К
inputs         А
к "К         Ае
D__inference_dense_11_layer_call_and_return_conditional_losses_571913]0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ }
)__inference_dense_11_layer_call_fn_571902P0в-
&в#
!К
inputs         А
к "К         ж
C__inference_dense_9_layer_call_and_return_conditional_losses_572205_011в.
'в$
"К
inputs         Ав
к "&в#
К
0         А
Ъ ~
(__inference_dense_9_layer_call_fn_572188R011в.
'в$
"К
inputs         Ав
к "К         Аи
F__inference_dropout_10_layer_call_and_return_conditional_losses_572220^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ и
F__inference_dropout_10_layer_call_and_return_conditional_losses_572232^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ А
+__inference_dropout_10_layer_call_fn_572210Q4в1
*в'
!К
inputs         А
p 
к "К         АА
+__inference_dropout_10_layer_call_fn_572215Q4в1
*в'
!К
inputs         А
p
к "К         Аи
F__inference_dropout_11_layer_call_and_return_conditional_losses_572279^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ и
F__inference_dropout_11_layer_call_and_return_conditional_losses_572291^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ А
+__inference_dropout_11_layer_call_fn_572269Q4в1
*в'
!К
inputs         А
p 
к "К         АА
+__inference_dropout_11_layer_call_fn_572274Q4в1
*в'
!К
inputs         А
p
к "К         А╖
E__inference_dropout_9_layer_call_and_return_conditional_losses_572150n<в9
2в/
)К&
inputs         		А
p 
к ".в+
$К!
0         		А
Ъ ╖
E__inference_dropout_9_layer_call_and_return_conditional_losses_572162n<в9
2в/
)К&
inputs         		А
p
к ".в+
$К!
0         		А
Ъ П
*__inference_dropout_9_layer_call_fn_572140a<в9
2в/
)К&
inputs         		А
p 
к "!К         		АП
*__inference_dropout_9_layer_call_fn_572145a<в9
2в/
)К&
inputs         		А
p
к "!К         		Ам
E__inference_flatten_3_layer_call_and_return_conditional_losses_572173c8в5
.в+
)К&
inputs         		А
к "'в$
К
0         Ав
Ъ Д
*__inference_flatten_3_layer_call_fn_572167V8в5
.в+
)К&
inputs         		А
к "К         Ав╕
D__inference_lambda_3_layer_call_and_return_conditional_losses_571931p?в<
5в2
(К%
inputs         KK

 
p 
к "-в*
#К 
0         KK
Ъ ╕
D__inference_lambda_3_layer_call_and_return_conditional_losses_571939p?в<
5в2
(К%
inputs         KK

 
p
к "-в*
#К 
0         KK
Ъ Р
)__inference_lambda_3_layer_call_fn_571918c?в<
5в2
(К%
inputs         KK

 
p 
к " К         KKР
)__inference_lambda_3_layer_call_fn_571923c?в<
5в2
(К%
inputs         KK

 
p
к " К         KK;
__inference_loss_fn_0_572302*в

в 
к "К ;
__inference_loss_fn_1_5723130в

в 
к "К ;
__inference_loss_fn_2_5723242в

в 
к "К я
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_569634ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_10_layer_call_fn_569640СRвO
HвE
CК@
inputs4                                    
к ";К84                                    я
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_569646ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_11_layer_call_fn_569652СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_569622ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_9_layer_call_fn_569628СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ┼
H__inference_sequential_3_layer_call_and_return_conditional_losses_571602y&'()*+,-./0123?в<
5в2
(К%
inputs         KK
p 

 
к "&в#
К
0         А
Ъ ┼
H__inference_sequential_3_layer_call_and_return_conditional_losses_571706y&'()*+,-./0123?в<
5в2
(К%
inputs         KK
p

 
к "&в#
К
0         А
Ъ ╬
H__inference_sequential_3_layer_call_and_return_conditional_losses_571789Б&'()*+,-./0123GвD
=в:
0К-
lambda_3_input         KK
p 

 
к "&в#
К
0         А
Ъ ╬
H__inference_sequential_3_layer_call_and_return_conditional_losses_571893Б&'()*+,-./0123GвD
=в:
0К-
lambda_3_input         KK
p

 
к "&в#
К
0         А
Ъ е
-__inference_sequential_3_layer_call_fn_571420t&'()*+,-./0123GвD
=в:
0К-
lambda_3_input         KK
p 

 
к "К         АЭ
-__inference_sequential_3_layer_call_fn_571453l&'()*+,-./0123?в<
5в2
(К%
inputs         KK
p 

 
к "К         АЭ
-__inference_sequential_3_layer_call_fn_571486l&'()*+,-./0123?в<
5в2
(К%
inputs         KK
p

 
к "К         Ае
-__inference_sequential_3_layer_call_fn_571519t&'()*+,-./0123GвD
=в:
0К-
lambda_3_input         KK
p

 
к "К         А╡
$__inference_signature_wrapper_570819М&'()*+,-./0123Cв@
в 
9к6
4
input_1)К&
input_1         KK"3к0
.
output_1"К
output_1         